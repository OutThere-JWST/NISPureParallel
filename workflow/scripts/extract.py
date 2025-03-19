#! /usr/bin/env python

# Import packages
import gc
import os
import yaml
import glob
import warnings
import argparse
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, join
from regions import Regions, PixCoord, EllipsePixelRegion

# Import grizli
import grizli
from grizli import multifit

# Silence warnings
warnings.filterwarnings('ignore')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Print version and step
    print(f'Extracting {fname}')
    print(f'grizli:{grizli.__version__}')

    # Skip if in a stellar field
    sfields = ['LMC', 'M31', 'M87', 'M101']
    with open('resources/aliases.yaml', 'r') as file:
        aliases = yaml.safe_load(file)
    if fname in aliases and np.logical_or.reduce(
        [f == aliases[fname] for f in sfields]
    ):
        print('Stellar field, skipping')
        return

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    home = os.path.join(fields, fname)

    # Subdirectories
    extract = os.path.join(home, 'Extractions')
    os.chdir(extract)

    # Skip if no GrismFLT files
    grism_files = glob.glob('*GrismFLT.fits')
    if len(grism_files) == 0:
        print('No GrismFLT files found')
        return

    # Load GroupFLT
    grp = multifit.GroupFLT(
        grism_files=glob.glob('*GrismFLT.fits'),
        catalog=f'{fname}-ir.cat.fits',
        cpu_count=ncpu,
        sci_extn=1,
        pad=800,
    )

    # Read extraction depth
    extract_mag = fits.getdata(os.path.join(extract, 'extract_mag.fits'))[0]

    # Get IDs
    cat = Table.read(f'{fname}-ir.cat.fits')
    cat = Table(cat, masked=True, copy=False)
    mag = cat['MAG_AUTO'].filled(np.inf)
    ids = cat['NUMBER'][mag <= extract_mag]

    # Define the extract_id function within main
    def extract_id(i, grism_files, catalog, fname):
        try:
            # Get beams
            beams = grp.get_beams(i, size=32, min_mask=0, min_sens=0.01)
            if not beams:
                return None

            # Extract using MultiBeam
            mb = multifit.MultiBeam(
                beams, fcontam=0.1, min_sens=0.01, min_mask=0, group_name=fname
            )
            mb.write_master_fits()

            # Free memory
            del mb, beams
            gc.collect()

            return i

        except Exception as e:
            print(f'Error processing ID {i}: {e}')
            return None

    # Multiprocess using ThreadPoolExecutor to avoid memory issues
    if ncpu > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(ncpu) as executor:
            extracted = list(executor.map(extract_id, ids))

    # Single process
    else:
        extracted = [extract_id(i) for i in ids]

    # Write catalog of extracted objects
    extracted = Table(
        [sorted([e for e in extracted if e is not None])], names=['NUMBER']
    )
    extracted.add_column(fname, name='field', index=0)
    extracted.write(f'{fname}-extracted.fits', overwrite=True)

    # Write DS9 region file
    extracted = join(extracted, cat, keys='NUMBER')
    Regions(
        [
            EllipsePixelRegion(
                PixCoord(c['X_IMAGE'], c['Y_IMAGE']),
                width=c['A_IMAGE'],
                height=c['B_IMAGE'],
                angle=c['THETA_IMAGE'] * u.rad,
                visual={'color': '#E20134', 'linewidth': 2},
            )
            for c in extracted
        ]
    ).write(f'{fname}-extracted.reg', format='ds9', overwrite=True)


if __name__ == '__main__':
    main()
