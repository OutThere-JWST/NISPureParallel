#! /usr/bin/env python

# Import packages
import os
import glob
import shutil
import argparse
import warnings
from multiprocessing import Pool

import numpy as np

# Packages for Plotting
import shapely
import spherely as sph

# Astropy packages
from astropy.io import fits
from grizli.aws import visit_processor
from matplotlib import pyplot, patches
from grizli.prep import visit_grism_sky
from astropy.table import Table
from astropy.io.fits import getdata, getheader
from grizli.pipeline import auto_script
from astropy.coordinates import Angle

# grizli packages
import grizli
from grizli import utils, jwst_utils

# Silence warnings
warnings.filterwarnings('ignore')


# Process image
def process_image(f, raw):
    # Copy raw file
    f = shutil.copy(f, raw)

    # Initialize image (flat-fielding, etc) and set correct header keywords
    jwst_utils.set_jwst_to_hst_keywords(f, oneoverf_correction=False, reset=True)


# Create main function
def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Print grizli and jwst versions
    print(f'Preprocessing {fname}')
    print(f'grizli:{grizli.__version__}')

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    prods = Table.read(os.path.join(fields, 'fields.fits'), fname)
    home = os.path.join(fields, fname)

    # Subdirectories
    rate = os.path.join(home, 'RATE')
    raw = os.path.join(home, 'RAW')
    prep = os.path.join(home, 'Prep')
    plots = os.path.join(home, 'Plots')
    persist = os.path.join(home, 'Persistence')
    extract = os.path.join(home, 'Extractions')
    for d in [raw, prep, plots, persist, extract]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.mkdir(d)

    # Plot field in context
    plot_field(fname, fields, plots)

    # Change to working directory
    os.chdir(fields)

    # Initialize images
    files = sorted(
        [os.path.join(rate, f).replace('uncal', 'rate') for f in prods['filename']]
    )

    # Multiprocess
    if ncpu == 1:
        for f in files:
            print(f)
            process_image(f, raw)
    else:
        with Pool(ncpu) as pool:
            pool.starmap(process_image, [(f, raw) for f in files])

    # Parse Visits
    visits, all_groups, info = auto_script.parse_visits(field_root=fname, RAW_PATH=raw)

    # Plot visits
    plot_visits(visits, fname, plots)

    # Subtract background from direct images
    # os.chdir(raw)
    # for group in all_groups:
    #     if 'direct' in group:  #
    #         visit_grism_sky(grism=group['grism'], column_average=False, ignoreNA=True)
    # os.chdir(fields)

    # Make visit associations
    assoc = info['EXPSTART', 'EXPTIME', 'INSTRUME']
    assoc['filter'] = ['-'.join(f.split('-')[:0:-1]) for f in info['FILTER']]

    # Find footprints
    footprints = []
    for f in sorted(glob.glob(f'{raw}/*rate.fits')):
        for v in visits:
            if os.path.basename(f) in v['files']:
                footprints.append(
                    v['footprints'][v['files'].index(os.path.basename(f))]
                )
                break

    # Fill out association table
    assoc['t_max'] = [int(getheader(f, 1)['MJD-END']) for f in files]
    assoc['proposal_id'] = [int(getheader(f, 0)['PROGRAM']) for f in files]
    assoc['footprint'] = footprints
    assoc['dataURL'] = [
        f'mast:JWST/product/{os.path.split(f)[1].replace("uncal", "rate")}'
        for f in files
    ]

    # Rename columns
    assoc.rename_columns(
        ['EXPSTART', 'EXPTIME', 'INSTRUME'], ['t_min', 'exptime', 'instrument_name']
    )

    # Visit process arguements
    prep_args = {
        'tweak_max_dist': 5,
        'oneoverf_kwargs': None,
        'snowball_kwargs': None,
        'imaging_bkg_params': None,
        'column_average': False,
    }

    # Other arguements
    other_args = {'is_parallel_field': True}

    # Process visit
    visit_processor.ROOT_PATH = fields
    visit_processor.process_visit(
        fname,
        tab=assoc,
        prep_args=prep_args,
        other_args=other_args,
        clean=False,
        sync=False,
        with_db=False,
        visit_split_shift=2,
        combine_same_pa=True,
    )

    # Plot Science Images
    files = glob.glob(os.path.join(prep, '*drz_sci.fits'))
    for file in files:
        # Split path
        f = os.path.split(file)[1]

        # Create figure
        fig, ax = pyplot.subplots(1, 1, figsize=(8, 8))

        # Get image data
        data = getdata(file)

        # Scaling
        vm = np.nanpercentile(data, [5, 95])

        # Plot images
        ax.imshow(data, vmin=-0.1 * vm[1], vmax=vm[1], cmap='gray_r')

        # Set axis
        ax.set_title(f)
        ax.axis('off')
        ax.set_aspect(1)

        # Save and close
        fig.savefig(
            os.path.join(plots, f.replace('.fits', '.pdf')), bbox_inches='tight'
        )
        pyplot.close(fig)


# Plot Field in context
def plot_field(fname, fields, plots):
    # Plot the field in context
    hdul = fits.open(os.path.join(fields, 'fields.fits'))

    # Create figure
    fig, ax = pyplot.subplots(figsize=(12, 12))

    # Plot current region in red
    ra, dec = plot_shapely(
        create_shapely(Table(hdul[fname].data)), ax, ec='#009F81', fc='#00FCCF'
    )

    # Plot all regions (except region we are on) in gray
    for hdu in hdul[1:]:
        if hdu.name == fname:
            continue
        plot_shapely(create_shapely(Table(hdu.data)), ax, ec='k', fc='gray')

    # Axis labels and limits
    ra_cen, dec_cen = (np.max(ra) + np.min(ra)) / 2, (np.max(dec) + np.min(dec)) / 2
    scale = np.cos(np.deg2rad(dec_cen))  # Scale for RA
    pad = 1 / 6  # Total padding along one axis in degrees (10 arcmin)
    δra, δdec = (np.max(ra) - np.min(ra) + pad) / scale, np.max(dec) - np.min(dec) + pad
    if δra > δdec:
        δdec = δra * scale
    else:
        δra = δdec / scale
    ax.set(xlabel='Right Ascension (ICRS)', xlim=(ra_cen + δra / 2, ra_cen - δra / 2))
    ax.set(ylabel='Declination (ICRS)', ylim=(dec_cen - δdec / 2, dec_cen + δdec / 2))
    ax.set(title=fname.lower().replace('-', r'$-$'))

    # Format labels correctly
    ax.xaxis.set_major_formatter(
        lambda x, _: Angle(x, unit='deg').to_string(
            unit='hour', sep=[r'$^\textrm{' + s + '}$' for s in 'hms']
        )
    )
    ax.tick_params(axis='x', labelrotation=25)
    ax.yaxis.set_major_formatter(
        lambda x, _: Angle(x, unit='deg')
        .to_string(unit='degree', sep=(r'$^\circ$', r"$'$", r"$''$"))
        .replace('-', r'$-$')
    )

    # Add grid
    ax.grid(True, which='major', ls='--', color='k', alpha=0.5)
    ax.grid(True, which='minor', ls=':', color='k', alpha=0.25)

    # Save figure
    fig.savefig(os.path.join(plots, f'{fname}-region.pdf'))
    pyplot.close(fig)


# Plot Visits
def plot_visits(visits, fname, plots):
    # Get color cycle
    ls_dic = {'CLEAR': '-', 'GR150R': '--', 'GR150C': ':'}
    color_dic = {
        'F090W': '#FFC33B',
        'F115W': '#FF6E3A',
        'F140M': '#FF5AAF',
        'F150W': '#E20134',
        'F158M': '#9F0162',
        'F200W': '#A40122',
    }

    # Create figure
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 12))

    # Enumerate visits
    ras, decs, fgs = [], [], []
    for i, v in enumerate(visits):
        # Get region box
        sr = utils.SRegion(v['footprint'])
        for coords in sr.xy:
            ra, dec = coords.T
            ras.append(ra)
            decs.append(dec)

        # Get filter-grism combo
        f, g = v['product'].split('-')[-2:]
        fgs.append(f'{f}-{g}')

        # Place patches for region
        for patch in sr.patch(
            ec=color_dic[f], fc='None', alpha=0.5, lw=3, ls=ls_dic[g]
        ):
            ax.add_patch(patch)

    # Concatenate coordinates
    ra = np.concatenate(ras)
    dec = np.concatenate(decs)

    # Add legend for fgs
    for fg in set(fgs):
        f, g = fg.split('-')
        ax.plot([], [], color=color_dic[f], ls=ls_dic[g], label=fg)
    ax.legend(fontsize=20, frameon=True)

    # Set axis parameters
    ra_cen, dec_cen = (np.max(ra) + np.min(ra)) / 2, (np.max(dec) + np.min(dec)) / 2
    scale = np.cos(np.deg2rad(dec_cen))
    pad = 1 / 360  # Total padding along one axis in degrees (10 arcsec)
    δra, δdec = (np.max(ra) - np.min(ra) + pad) / scale, np.max(dec) - np.min(dec) + pad
    if δra > δdec:
        δdec = δra * scale
    else:
        δra = δdec / scale
    ax.set(xlabel='Right Ascension (ICRS)', xlim=(ra_cen + δra / 2, ra_cen - δra / 2))
    ax.set(ylabel='Declination (ICRS)', ylim=(dec_cen - δdec / 2, dec_cen + δdec / 2))
    ax.set(title=fname.lower().replace('-', r'$-$'))

    # Format labels correctly
    ax.xaxis.set_major_formatter(
        lambda x, _: Angle(x, unit='deg').to_string(
            unit='hour', sep=[r'$^\textrm{' + s + '}$' for s in 'hms']
        )
    )
    ax.tick_params(axis='x', labelrotation=25)
    ax.yaxis.set_major_formatter(
        lambda x, _: Angle(x, unit='deg')
        .to_string(unit='degree', sep=(r'$^\circ$', r"$'$", r"$''$"))
        .replace('-', r'$-$')
    )

    # Add grid
    ax.grid(True, which='major', ls='--', color='k', alpha=0.5)
    ax.grid(True, which='minor', ls=':', color='k', alpha=0.25)

    # Save figure
    fig.savefig(os.path.join(plots, f'{fname}-visits.pdf'), bbox_inches='tight')
    pyplot.close(fig)


def create_shapely(field):
    """Create a region from a row of the table."""

    regions = []
    for row in field:
        # Get the target ra, dec and v3 pa
        ra, dec = row['targ_ra'], row['targ_dec']
        pa = -row['gs_v3_pa'] * np.pi / 180  # Convert to radians

        # Create the unite square and rotate it
        side_length = 1.1 / 60  # Degrees
        x = side_length * np.array([-1, 1, 1, -1])
        y = side_length * np.array([-1, -1, 1, 1])
        xr = x * np.cos(pa) - y * np.sin(pa)
        yr = x * np.sin(pa) + y * np.cos(pa)
        xr /= np.cos(np.deg2rad(dec))
        x, y = xr + ra, yr + dec

        regions.append(sph.create_polygon(np.array([x, y]).T))

    region = regions[0]
    for i, r in enumerate(regions[1:]):
        region = sph.union(region, r)

    return shapely.from_wkt(sph.to_wkt(region))


# Plot Shapely Object
def plot_shapely(r, ax, ec='r', fc='r'):
    # If MultiPolygon, plot each
    if hasattr(r, 'geoms'):
        # Loop over geoms
        ras, decs = [], []  # Keep track of coordinates
        for g in r.geoms:
            # Plot each geom
            ra, dec = plot_shapely(g, ax, ec=ec, fc=fc)
            ras.append(ra)
            decs.append(dec)

        # Return list of all coordinates
        return np.concatenate(ras), np.concatenate(decs)

    if r.geom_type == 'LineString':
        return [], []

    # Get ra,dec of exterior
    coords = np.array(r.exterior.xy)

    # Plot main polygon in red
    patch = patches.Polygon(coords.T, closed=True, edgecolor=ec, facecolor=fc, lw=3)
    ax.add_patch(patch)

    # Return ra,dec
    return coords


# Run main function
if __name__ == '__main__':
    main()
