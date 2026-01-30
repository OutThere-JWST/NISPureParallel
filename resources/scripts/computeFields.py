#! /usr/bin/env python

import os
import argparse
import multiprocessing as mpl
from concurrent.futures import ThreadPoolExecutor

import yaml
import numpy as np
import spherely as sph
from tqdm import tqdm, trange
from astropy.io import fits
from astropy.table import join, vstack
from astroquery.mast import MastMissions
from astropy.coordinates import SkyCoord, get_constellation

# Query MAST for JWST data
missions = MastMissions(mission='jwst')


def create_region(row):
    """Create a region from a row of the table."""
    # Get the target ra, dec and v3 pa
    ra, dec = row['targ_ra'], row['targ_dec']
    pa = -row['gs_v3_pa'] * np.pi / 180  # Convert to radians

    # Create the unit square and rotate it
    side_length = 1.1 / 60  # Degrees
    x = side_length * np.array([-1, 1, 1, -1])
    y = side_length * np.array([-1, -1, 1, 1])
    xr = x * np.cos(pa) - y * np.sin(pa)
    yr = x * np.sin(pa) + y * np.cos(pa)
    xr /= np.cos(np.deg2rad(dec))
    x, y = xr + ra, yr + dec

    return sph.create_polygon(np.array([x, y]).T)


def get_products(field):
    # Loop as MAST sometimes doesn't return all products
    prods = []
    while len(prods) != len(field):
        # Get all products for the field
        allprods = missions.get_product_list(field)

        # Filter for 1b products
        prods = missions.filter_products(allprods, category='1b', file_suffix='_uncal')

    # Remove overlapping columns
    field.remove_columns(['access', 'category'])

    # Rename product column
    prods.rename_column('dataset', 'fileSetName')

    return join(field, prods)


if __name__ == '__main__':
    # Get list of proposal ids from the user
    parser = argparse.ArgumentParser(description='Compute Fields for JWST NIRISS')
    parser.add_argument(
        '--separation',
        type=float,
        help='Maximum separation between fields (arcsec)',
        default=30,
    )
    parser.add_argument(
        '--ignore-ids',
        type=str,
        nargs='+',
        help='List of proposal IDs to ignore',
        default=[
            1085,  # NIRISS Focus Sweep
            1089,  # NIRISS Grism Flux Cal
            1090,  # NIRISS Grism Wave Cal
            # 1202,  # Survey of Reflection Nebula
            4477,  # NIRISS Grism Contam Cal
        ],
    )
    parser.add_argument(
        '--maxcpu',
        type=int,
        help='Maximum number of CPUs to use for multiprocessing',
        default=mpl.cpu_count(),
    )

    # Parse arguements
    args = parser.parse_args()
    separation = args.separation
    ignore_ids = args.ignore_ids

    # Columns, remove problematic ones (seems to be a bug in MAST)
    columns = list(missions.get_column_list()['name'])
    badcols = ['effexptm', 'gsstrttm', 'gsendtim', 'texptime']

    # Query for prime and pure parallel
    print('Querying MAST (Pure Parallel and Prime)...')
    wfss_prime = missions.query_criteria(
        template='NIRISS Wide Field Slitless Spectroscopy',
        select_cols=[c for c in columns if c not in badcols],
        instrume='NIRISS',
        productLevel='1*',
        limit=10000,
    )

    # Get Coordinated Parallel Observations
    print('Querying MAST (Coordinated Parallel)...')
    nis_parallel = missions.query_criteria(
        select_cols=[c for c in columns if c not in badcols],
        instrume='NIRISS',
        productLevel='1*',
        limit=10000,
        expripar='PARALLEL_COORDINATED',
    )

    # Limit to only the WFSS observations in wfss_parallel
    # Find programs with at least 1 grism
    has_grism = np.logical_or.reduce(
        [nis_parallel['filter'] == f'GR150{g}' for g in ['R', 'C']]
    )
    wfss_parallel = nis_parallel[
        np.logical_or.reduce(
            [
                nis_parallel['program'] == p
                for p in np.unique(nis_parallel[has_grism]['program'])
            ]
        )
    ]

    # Combine prime and parallel WFSS observations
    wfss = vstack([wfss_prime, wfss_parallel])

    # Remove ignored proposal IDs
    remove = np.logical_or.reduce([wfss['program'] == p for p in ignore_ids])
    wfss = wfss[~remove]
    print(f'Found {len(wfss)} observations')

    # Remove empty columns
    wfss.remove_columns([c for c in wfss.colnames if wfss.mask[c].sum()])

    # Order by observation date
    wfss.sort('date_obs')

    # Compute primary ID
    wfss['primary_id'] = [int(o[1:6]) for o in wfss['obs_id']]

    # Create regions for all observations
    print('Breaking into invidual observations')
    obs = [wfss[i : i + 1] for i in trange(len(wfss))]
    regs = [create_region(row) for row in wfss]

    # Combine all overlapping fields
    fields = [obs[0]]
    f_regs = [regs[0]]
    obs = obs[1:]
    regs = regs[1:]

    # Iterate until all regions have been assigned to a field
    print('Combining Nearby Fields')
    pbar = tqdm(total=len(regs))
    while len(regs) > 0:
        # Iterate over regions
        for i, r in enumerate(regs):
            # If most recent field is within separation distance
            if sph.distance(f_regs[-1], r, radius=180 / np.pi) < separation / 3600:
                # Combine regions and observations
                f_regs[-1] = sph.union(f_regs[-1], regs.pop(i))
                fields[-1] = vstack([fields[-1], obs.pop(i)])
                pbar.update(1)

                # Break loop and start over
                notfinished = False
                break

            # Finished loop with no intersection
            notfinished = True

        # If no intersection, add new fields
        if notfinished:
            f_regs.append(regs.pop(0))
            fields.append(obs.pop(0))
            pbar.update(1)
    pbar.close()

    # Multiprocess to get products for each field
    print('Getting products for each field...')
    with ThreadPoolExecutor(max_workers=args.maxcpu) as executor:
        results = list(tqdm(executor.map(get_products, fields), total=len(fields)))

    # Load names if it exists
    if os.path.isfile('resources/names.yaml'):
        with open('resources/names.yaml', 'r') as f:
            names = yaml.safe_load(f)
    else:
        names = {}

    # Iterate over results to assign names
    for result in results:
        # Get first observation in the field
        r = result[0]

        # Get unique key for the field
        key = str(r['obs_id'])

        # If the field already has a name, skip it
        if key in names:
            continue

        # Get constellation name
        con = get_constellation(
            SkyCoord(*r['targ_ra', 'targ_dec'], unit='deg'), short_name=True
        ).lower()

        # Find instance of constellation
        j = 0
        while f'{con}-{str(j).zfill(2)}' in names.values():
            j += 1
        name = f'{con}-{str(j).zfill(2)}'

        # Assign name to field
        names[key] = name

    # Save names to file
    with open('resources/names.yaml', 'w') as f:
        yaml.dump(names, f, default_flow_style=False, sort_keys=False)

    # Write field names to file
    if not os.path.isdir('FIELDS'):
        os.mkdir('FIELDS')

    # Save fields to FITS file
    print('Saving to FITS...')
    hdul = fits.HDUList([fits.PrimaryHDU()])
    field_names = []
    for i, f in enumerate(tqdm(results)):
        # Append field observations
        hdul.append(fits.BinTableHDU(f))
        field_name = names[str(f[0]['obs_id'])]
        hdul[-1].header['EXTNAME'] = field_name
        field_names.append(field_name)

    # Save FITS file
    hdul.writeto('FIELDS/fields.fits', overwrite=True)

    # Write field names
    # with open('FIELDS/fields.txt', 'w') as f:
    #     f.write('\n'.join(field_names))
    # print('Done! Fields saved to FIELDS/fields.fits and FIELDS/fields.txt')
