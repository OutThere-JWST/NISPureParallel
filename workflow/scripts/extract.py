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
from grizli import multifit

# Extract beams
def extractBeams(b,root):

    if len(b) == 0: 
        return

    # Extract beams
    mb = multifit.MultiBeam(b, fcontam=0.1, min_sens=0.01, min_mask=0, group_name=root)
    mb.write_master_fits()
    
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
    fields = os.path.join(main,'CLUSTERS')
    home = os.path.join(fields,cname)
    print(f'Extracting {cname}')

    # Subdirectories
    logs = os.path.join(home,'logs')
    plots = os.path.join(home,'Plots')
    extract = os.path.join(home,'Extractions')
    os.chdir(extract)

    # Redirect stdout and stderr to file
    if not args.verbose:
        sys.stdout = open(os.path.join(logs,'extr.out'),'w')
        sys.stderr = open(os.path.join(logs,'extr.err'),'w')

    # Print grizli and jwst versions
    print(f'grizli:{grizli.__version__}')

    # Load GroupFLT
    grp = multifit.GroupFLT(
        grism_files=glob.glob('*GrismFLT.fits'),
        catalog=f'{cname}-ir.cat.fits',
        cpu_count=ncpu, sci_extn=1, pad=800
    )

    # # Move files
    # for f in glob(f'{root}_*{str(i).rjust(5,"0")}.*.png'):
    #     new = f.replace('full','zfit').replace('stack','twod')
    #     new = '.'.join(new.split('.')[-2:])
    #     os.rename(f,os.path.join(extract_plots,f'{str(i)}.{new}'))

    # # Create oned spectrum figure
    # fig = mb.oned_figure()
    # fig.savefig(os.path.join(extract_plots,f'{i}_oned.png'))
    # pyplot.close(fig)

    # # Create 2d spectrum figure
    # hdu,fig = mb.drizzle_grisms_and_PAs(size=38, scale=0.5, diff=True, kernel='square', pixfrac=0.5)
    # fig.savefig(os.path.join(extract_plots,f'{i}_twod.png'))
    # pyplot.close(fig)

    # Get IDs
    cat = Table.read(f'{cname}-ir.cat.fits')
    mag = cat['MAG_AUTO'].filled(np.inf)
    ids = cat['NUMBER'][mag < 27]

    # Crate arguements
    args = [(grp.get_beams(i, size=32, min_mask=0, min_sens=0.01),cname) for i in ids]

    # Multiprocessing pool
    pool = Pool(processes=ncpu)
    pool.starmap_async(extractBeams,args)
    pool.close()
    pool.join()