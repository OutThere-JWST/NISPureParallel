#! /usr/bin/env python

# Import packages
import os
import re
import glob
import math
import shutil
import argparse
import warnings
import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool
from scipy import stats, optimize
from reproject import reproject_interp

# Astropy packages
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import make_lupton_rgb

# grizli packages
import grizli
from grizli import utils
from grizli.aws import visit_processor
from grizli.pipeline import auto_script

# Silence warnings
warnings.filterwarnings('ignore')

# Set priority of filters
filt_ref = dict(
    F115W=['F115W', 'F150W', 'F200W'],
    F150W=['F150W', 'F200W'],
    F200W=['F200W'],
)


# Model contamination for filt
def model_contam(fname, filt, dirs, mags, projs, grism_files):
    # Unpack
    fields, prep, plots, extract = dirs  # directories
    min_mag, extract_mag = mags  # magnitudes
    projref, projsize = projs  # projection

    # Change directory
    os.chdir(prep)

    # Force detection image
    for f in filt_ref[filt]:
        obs = Table.read(os.path.join(fields, 'field-obs.fits'), fname)
        if f'CLEAR;{f}' in np.unique(obs['filters']):
            ref_filt = f
            break
    ref = f'{fname}-{ref_filt.lower()}n-clear_drc_sci.fits'

    # Create grism model
    grp = auto_script.grism_prep(
        field_root=fname,
        PREP_PATH=prep,
        EXTRACT_PATH=extract,
        refine_niter=3,
        refine_mag_limits=[min_mag - 1, extract_mag - 2],
        prelim_mag_limit=extract_mag,
        init_coeffs=[1, -0.6],
        pad=(1024, 1024),
        files=grism_files,
        model_kwargs={'compute_size': False, 'size': 48},
        subtract_median_filter=False,
        use_jwst_crds=False,
        force_ref=ref,
        cpu_count=1,
        refine_threads=1,
        # sep_background_kwargs={}
    )

    # Drizzle grism models (scale same as mosaic)
    grp.drizzle_grism_models(
        root=fname, kernel='square', scale=0.04, pixfrac=0.75, write_ctx=True
    )

    # Get angles
    pas = list(grp.PA[filt].keys())

    # Iterate over angles
    for pa in pas:
        # Load context map
        prefix = f'{fname}-{filt.lower()}-{pa}_grism'
        hdul = fits.open(os.path.join(extract, f'{prefix}_ctx.fits'))
        ctx, h = hdul[0].data, hdul[0].header

        # Make exposure time map
        exptime = np.zeros(ctx.shape)
        for i in range(math.floor(np.log2(ctx.max())) + 1):  # Iterate over bits
            # Get exposure time of file
            file = os.path.join(prep, h[f'FLT{str(i+1).zfill(5)}'])
            t = fits.getval(file, 'EXPTIME')

            # Set exposure time if bit i is set in ctx
            exptime[np.bitwise_and(ctx, 2**i) > 0] += t

        # Reproject exposure time map
        proj, _ = reproject_interp((exptime, h), projref, projsize)

        # Save exposure time map
        fits.PrimaryHDU(proj, header=h).writeto(
            os.path.join(extract, f'{prefix}_exp.fits'), overwrite=True
        )

        # Copy grism model plots
        print(f'Copying {prefix} plots')
        shutil.copy(f'{prefix}.png', plots)

    return pas


def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Print version and step
    print(f'Modelling Contamination {fname}')
    print(f'grizli:{grizli.__version__}')

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    home = os.path.join(fields, fname)

    # Load observations (direct vs grism)
    obs = Table.read(os.path.join(fields, 'field-obs.fits'), fname)
    obs['filters'] = [re.sub(r'150[RC]', '', f) for f in obs['filters']]

    # Subdirectories
    prep = os.path.join(home, 'Prep')
    plots = os.path.join(home, 'Plots')
    extract = os.path.join(home, 'Extractions')

    # Go to prep directory
    os.chdir(prep)

    # Make a table with file information
    files = sorted(glob.glob('*rate.fits'))
    res = visit_processor.res_query_from_local(files=files)
    is_grism = np.array(['GR' in filt for filt in res['filter']])
    un = utils.Unique(res['pupil'])

    # Skip if no grism
    if not np.any(is_grism):
        print('No grism files, skipping')
        return

    # Projection Reference header
    with fits.open(os.path.join(prep, f'{fname}-ir_drc_sci.fits')) as f:
        projref = WCS(f[0].header)
        projsize = f[0].data.shape

    # Determine catalog depth
    cat = Table.read(os.path.join(prep, f'{fname}-ir.cat.fits'))
    cat = Table(cat, masked=True, copy=False)
    mag = cat['MAG_AUTO'][np.invert(cat['MAG_AUTO'].mask)]
    kde = stats.gaussian_kde(mag)  # KDE Estimate
    mode_mag = optimize.minimize(lambda x: -kde(x), np.median(mag)).x  # Modes

    # Determine exposure time offset
    times = {
        f: (obs['t_exptime'][obs['filters'] == f]).sum()
        for f in np.unique(obs['filters'])
    }
    t_clear = np.sum([times[f] for f in times if 'CLEAR' in f])  # Total Direct
    t_grism = np.max([times[f] for f in times if 'GR' in f])  # Max Grism
    offset = 2.5 * np.log10(np.sqrt(t_grism / t_clear))  # Scales with sqrt(t)

    # Determine extraction depth
    extract_mag = mode_mag + offset - 1.5  # 1.5 mag fainter than mode

    # Save extraction depth
    fits.PrimaryHDU(data=[extract_mag]).writeto(
        os.path.join(extract, 'extract_mag.fits'), overwrite=True
    )

    # Create multiprocess arguments
    args = [
        (
            fname,
            filt,
            (fields, prep, plots, extract),
            (mode_mag, extract_mag),
            (projref, projsize),
            ['{dataset}_rate.fits'.format(**row) for row in res[is_grism & un[filt]]],
        )
        for filt in filt_ref
        if np.any(is_grism & un[filt])  # Only if grism files are available
    ]

    # Multiprocess over filters
    with Pool(ncpu) as pool:
        all_pas = pool.starmap(model_contam, args)
    unique_pas = np.unique(np.concatenate(all_pas))  # Unique PAs

    # Change to extract directory
    os.chdir(extract)

    # Iterate over suffixes
    for end in ['sci', 'clean']:
        # Iterate over unique grisms
        for pa in unique_pas:
            print(f'Processing {end} RGB for angle {pa}')

            # Get filters for this grism
            all_filts = ['f200w', 'f150w', 'f115w']
            files = [f'{fname}-{f}-{pa}_grism_{end}.fits' for f in all_filts]

            # Reproject grism images (zero if not available)
            ims = []
            for file in files:
                if os.path.exists(file):
                    im, _ = reproject_interp(file, projref, projsize)
                    fits.PrimaryHDU(im).writeto(
                        file.replace('.fits', '_proj.fits'), overwrite=True
                    )
                else:
                    im = np.zeros(projsize)
                ims.append(im)

            # Make RGB
            filename = os.path.join(plots, f'{fname}-grism.{pa}_{end}.png')
            rgb = make_lupton_rgb(*ims, filename=None, Q=20, stretch=0.1)

            # Transform to use different colors to be colorblind friendly
            transform = np.array(
                [
                    [246, 2, 57],  # Carmine
                    [255, 110, 58],  # Burning Orange
                    [255, 172, 59],  # Frenzee
                ]
            )  # 255,220,61
            rgb = np.dot(rgb, transform / 255).clip(0, 255).astype(np.uint8)

            # Get diemsnions
            xsize = 8
            ny, nx, _ = rgb.shape
            dpi = int(nx / xsize)
            xsize = nx / dpi
            dim = [xsize, xsize / nx * ny]

            # Create output figure
            fig, ax = pyplot.subplots(figsize=dim, dpi=dpi)
            ax.axis('off')
            ax.imshow(rgb, origin='lower', extent=(-nx / 2, nx / 2, -ny / 2, ny / 2))

            # Add labels
            for i, f in enumerate(all_filts):
                ax.text(
                    0.03 + 0.1 * i,
                    0.97,
                    f.upper(),
                    color=transform[i] / 255,
                    bbox=dict(facecolor='w', alpha=1),
                    size=14,
                    ha='left',
                    va='top',
                    transform=ax.transAxes,
                )

            # Save figure
            fig.tight_layout(pad=0)
            fig.savefig(filename, pad_inches=0, bbox_inches='tight')
            pyplot.close(fig)


if __name__ == '__main__':
    main()
