#! /usr/bin/env python

# Import packages
import os
import sys
import json
import glob
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from multiprocessing import Pool

# Silence warnings
warnings.filterwarnings('ignore')

# Import grizli
import grizli
from grizli import fitting
from grizli.pipeline import auto_script

# Fit Redshift
def zfit(i):

    # Fit
    fitting.run_all_parallel(i,zr=[0.1,5],args_file='fit_args.npy',verbose=True)

    # # Create oned spectrum figure
    # fig = mb.oned_figure()
    # fig.savefig(os.path.join(extract_plots,f'{i}_oned.png'))
    # pyplot.close(fig)

    # # Create 2d spectrum figure
    # hdu,fig = mb.drizzle_grisms_and_PAs(size=38, scale=0.5, diff=True, kernel='square', pixfrac=0.5)
    # fig.savefig(os.path.join(extract_plots,f'{i}_twod.png'))
    # pyplot.close(fig)

if __name__ == '__main__':

    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int,default=1)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main,'FIELDS')
    home = os.path.join(fields,fname)
    print(f'Fitting {fname}')

    # Subdirectories
    logs = os.path.join(home,'logs')
    plots = os.path.join(home,'Plots')
    extract = os.path.join(home,'Extractions')
    os.chdir(extract)

    # Redirect stdout and stderr to file
    if not args.verbose:
        sys.stdout = open(os.path.join(logs,'zfit.out'),'w')
        sys.stderr = open(os.path.join(logs,'zfit.err'),'w')

    # Print grizli and jwst versions
    print(f'grizli:{grizli.__version__}')

    # Generate fit arguements
    pline={
        'kernel':'square',
        'pixfrac':0.5,
        'pixscale':0.04,
        'size':8,
        'wcs':None
        }
    auto_script.generate_fit_params(
        pline=pline,field_root=fname,min_sens=0.01,min_mask=0.0
        )

    # Get IDs
    cat = Table.read(f'{fname}-ir.cat.fits')
    mag = cat['MAG_AUTO'].filled(np.inf)
    ids = cat['NUMBER'][mag < 24]

    # Multiprocessing pool
    with Pool(processes=ncpu) as pool:
        result = pool.map_async(zfit,ids)
        output = result.get()
        pool.close()
        pool.join()

    # # Move files
    # for f in glob(f'{root}_*{str(i).rjust(5,"0")}.*.png'):
    #     new = f.replace('full','zfit').replace('stack','twod')
    #     new = '.'.join(new.split('.')[-2:])
    #     os.rename(f,os.path.join(extract_plots,f'{str(i)}.{new}'))

    # Create fitting catalog
    # out = vstack([
    #     Table.read(f) for f in tqdm(glob(f'{root}*row.fits'))
    # ])
    # out[np.argsort(out['id'])].write(f'{root}_fitresults.fits',overwrite=True)