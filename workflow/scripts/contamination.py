#! /usr/bin/env python

# Import packages
import os
import sys
import glob
import json
import shutil
import argparse
import warnings
import numpy as np
from matplotlib import pyplot
from reproject import reproject_adaptive as reproject

# Astropy packages
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import make_lupton_rgb

# Silence warnings
warnings.filterwarnings('ignore')

# grizli packages
import grizli
from grizli import utils
from grizli.aws import visit_processor
from grizli.pipeline import auto_script

if __name__ == '__main__':

    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int,default=1)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    cname = args.fieldname
    ncpu = args.ncpu

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main,'FIELDS')
    home = os.path.join(fields,cname)
    print(f'Modelling Contamination {cname}')

    # Subdirectories
    logs = os.path.join(home,'logs')
    prep = os.path.join(home,'Prep')
    plots = os.path.join(home,'Plots')
    extract = os.path.join(home,'Extractions')

    # Redirect stdout and stderr to file
    if not args.verbose:
        sys.stdout = open(os.path.join(logs,'contam.out'),'w')
        sys.stderr = open(os.path.join(logs,'contam.err'),'w')

    # Go to prep directory
    os.chdir(prep)

    # Print grizli and jwst versions
    print(f'grizli:{grizli.__version__}')

    # Make a table with file information
    files = sorted(glob.glob('*rate.fits'))
    res = visit_processor.res_query_from_local(files=files)
    is_grism = np.array(['GR' in filt for filt in res['filter']])
    un = utils.Unique(res['pupil'])

    # Set priority of filters
    filt_ref = dict(
        F115W=['F115W','F150W','F200W'],
        F150W=['F150W','F200W'],
        F200W=['F200W'],
    )

    # Iterate over filters
    for filt in un.values:

        # Change directory
        os.chdir(prep)
        
        # Get grism files
        grism_files = ['{dataset}_rate.fits'.format(**row)
                        for row in res[is_grism & un[filt]]]

        # Force detection image
        for f in filt_ref[filt]:
            obs = Table.read(os.path.join(fields,'field-obs.fits'),cname)
            if f'CLEAR;{f}' in np.unique(obs['filters']):
                ref_filt = f
                break
        ref = f'{cname}-{ref_filt.lower()}n-clear_drc_sci.fits'

        # Create grism model
        grp = auto_script.grism_prep(
            field_root=cname,
            PREP_PATH=prep,
            EXTRACT_PATH=extract,
            refine_niter=3,
            refine_mag_limits=[16, 24],
            prelim_mag_limit=26,
            init_coeffs=[1, -0.6],
            pad=(1024, 1024),
            files=grism_files,
            model_kwargs={'compute_size': False, 'size':48},
            subtract_median_filter=False,
            use_jwst_crds=True,
            force_ref=ref,
            cpu_count=ncpu
            # sep_background_kwargs={}
            )

        # Drizzle grism models
        grp.drizzle_grism_models(root=cname, kernel='square', scale=0.04, pixfrac=0.75)

    # Copy grism model plots
    for f in glob.glob(os.path.join(extract,'*grism*png')): shutil.copy(f,plots)

    # Reference header
    with fits.open(os.path.join(prep,f'{cname}-ir_drc_sci.fits')) as f:
        ref = WCS(f[0].header)
        size = f[0].data.shape
        
    # Iterate over suffixes
    for end in ['sci','clean']: # For now, only sci
        suffix = f'_grism_{end}.fits'

        # Get files
        files = sorted(glob.glob(os.path.join(extract,f'{cname}-*{suffix}')))[::-1]

        # Get filters
        filters,grisms = [],[]
        for f in files:
            clean = f.replace(cname,'').replace(suffix,'').split('-')
            filters.append(clean[-2])
            grisms.append(clean[-1])
        filters,grisms = np.array(filters),np.array(grisms)

        # Iterate over unique grisms
        for grism in np.unique(grisms):

            print(f'Processing {end} RGB for angle {grism}')

            # Get filters for this grism
            gfilts = list(filters[grisms == grism])

            # Reproject grism images (zero if not available)
            all_filts = ['f200w','f150w','f115w']
            ims = [
                reproject(f'{cname}-{gf}-{grism}{suffix}',ref,size,conserve_flux=True)[0] if gf in gfilts else np.zeros(size)
                for gf in all_filts
                ]

            # Make RGB
            filename = os.path.join(plots,f'{cname}-grism.{grism}_{end}.png')
            rgb = make_lupton_rgb(*ims,filename=filename,Q=20,stretch=0.1)
            
            # Get diemsnions
            xsize = 8
            ny, nx, _ = rgb.shape
            dpi = int(nx/xsize)
            xsize = nx/dpi
            dim = [xsize, xsize/nx*ny]

            # Create output figure
            fig, ax = pyplot.subplots(figsize=dim, dpi=dpi)
            ax.axis('off')
            ax.imshow(rgb, origin='lower', extent=(-nx/2, nx/2, -ny/2, ny/2))

            # Add labels
            for i,f in enumerate(all_filts):
                ax.text(0.03+0.1*i, 0.97, f.upper(),color='rgb'[i],
                        bbox=dict(facecolor='w', alpha=1),
                        size=14, ha='left', va='top',
                        transform=ax.transAxes)

            # Save figure
            fig.tight_layout(pad=0)
            fig.savefig(filename,pad_inches=0,bbox_inches='tight')
            pyplot.close(fig)
