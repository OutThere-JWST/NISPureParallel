#! /usr/bin/env python

# Import packages
import os
import re
import yaml
import glob
import warnings
import argparse
import numpy as np
from astropy.table import Table
from multiprocessing import Pool
from scipy import stats, optimize

# Import grizli
import grizli
from grizli import multifit

# Silence warnings
warnings.filterwarnings('ignore')


# Extraction function
def extract_id(i, grp, fname):
    # Get beams from group
    beams = grp.get_beams(i, size=32, min_mask=0, min_sens=0.01)
    if len(beams) == 0:
        return  # Skip if no beams

    # Extract beam
    mb = multifit.MultiBeam(
        beams, fcontam=0.1, min_sens=0.01, min_mask=0, group_name=fname
    )
    mb.write_master_fits()

    # Keep track of extracted objects
    return i


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

    # Skip if in a stellar field
    sfields = ['LMC', 'M31', 'M101']
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

    # Load exposure times (direct vs grism)
    obs = Table.read(os.path.join(fields, 'field-obs.fits'), fname)
    obs['filters'] = [re.sub(r'150[RC]', '', f) for f in obs['filters']]

    # Subdirectories
    # plots = os.path.join(home, 'Plots')
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

    # Determine catalog depth
    cat = Table.read(os.path.join(extract, f'{fname}-ir.cat.fits'))
    mag = cat['MAG_AUTO'][np.invert(cat['MAG_AUTO'].mask)]
    kde = stats.gaussian_kde(mag)  # KDE Estimate
    mode_mag = optimize.minimize(lambda x: -kde(x), np.median(mag)).x  # Modes

    # Determine exposure time offset
    times = {
        f: (obs['t_exptime'][obs['filters'] == f]).sum()
        for f in np.unique(obs['filters'])
    }
    t_clear = np.sum([times[f] for f in times if 'CLEAR' in f])  # Total Direct
    t_grism = np.max([times[f] for f in times if 'GR' in f] + [0])  # Max Grism
    offset = 2.5 * np.log10(np.sqrt(t_grism / t_clear))  # Scales with sqrt(t)

    # Determine extraction depth
    extract_mag = mode_mag + offset - 1.5  # 1.5 mag fainter than mode

    # Get IDs
    cat = Table.read(f'{fname}-ir.cat.fits')
    mag = cat['MAG_AUTO'].filled(np.inf)
    ids = cat['NUMBER'][mag <= extract_mag]

    # Multiprocess
    with Pool(ncpu) as pool:
        # Iterate over IDs
        extracted = pool.starmap(extract_id, [(i, grp, fname) for i in ids])

    # Write catalog of extracted objects
    Table([[e for e in extracted if e is not None]], names=['NUMBER']).write(
        f'{fname}-extracted.fits', overwrite=True
    )


if __name__ == '__main__':
    main()
