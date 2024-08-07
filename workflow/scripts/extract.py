#! /usr/bin/env python

# Import packages
import os
import glob
import warnings
import argparse
import numpy as np
from astropy.table import Table

# Import grizli
import grizli
from grizli import multifit

# Silence warnings
warnings.filterwarnings('ignore')


def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Print version and step
    print(f'Extracting {fname}')
    print(f'grizli:{grizli.__version__}')

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    home = os.path.join(fields, fname)

    # Subdirectories
    # plots = os.path.join(home, 'Plots')
    extract = os.path.join(home, 'Extractions')
    os.chdir(extract)

    # Load GroupFLT
    grism_files = glob.glob('*GrismFLT.fits')
    if len(grism_files) == 0:
        print('No GrismFLT files found')
        return
    grp = multifit.GroupFLT(
        grism_files=glob.glob('*GrismFLT.fits'),
        catalog=f'{fname}-ir.cat.fits',
        cpu_count=ncpu,
        sci_extn=1,
        pad=800,
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
    cat = Table.read(f'{fname}-ir.cat.fits')
    mag = cat['MAG_AUTO'].filled(np.inf)
    ids = cat['NUMBER'][mag < 24]

    # Iterate over IDs
    extracted = []
    for i in ids:
        # Get beams from group
        beams = grp.get_beams(i, size=32, min_mask=0, min_sens=0.01)
        if len(beams) == 0:
            continue  # Skip if no beams

        # Extract beam
        mb = multifit.MultiBeam(
            beams, fcontam=0.1, min_sens=0.01, min_mask=0, group_name=fname
        )
        mb.write_master_fits()

        # Keep track of extracted objects
        extracted.append(i)

    # Write catalog of extracted objects
    Table([extracted], names=['NUMBER']).write(
        f'{fname}-extracted.fits', overwrite=True
    )


if __name__ == '__main__':
    main()
