#! /usr/bin/env python

# Python Packages
import os
import tqdm
import argparse
# import multiprocessing as mp

# Computational Packages
import numpy as np
from scipy.optimize import minimize

# Astropy Packages
from astropy.io import fits
from astropy.table import Table, vstack, join
from astroquery.mast import Observations, MastMissions
from astropy.coordinates import SkyCoord, get_constellation

# Geometry packages
from sregion import SRegion
from spherical_geometry.polygon import SingleSphericalPolygon

# Query MAST for JWST data
missions = MastMissions(mission='jwst')


# Parametric equation for an arc from 0 to 1
def parametric_arc(a, b):
    # Get angle between
    theta = np.arccos(np.dot(a, b))

    # Get perpendicular vector
    c = np.cross(np.cross(a, b), a)
    c /= np.sqrt(c.dot(c))  # Normalize to unit

    # Define arc function
    def arc(t):
        scaled_t = t * theta
        return a * np.cos(scaled_t) + c * np.sin(scaled_t)

    return arc


# Distance between two arcs (result in radians)
def distance_between_arcs(a, b, c, d):
    # If points are (essentially) the same, skip
    if np.dot(a, b) > 1 or np.dot(c, d) > 1:
        return np.inf

    # Get arcs from points
    arc1 = parametric_arc(a, b)
    arc2 = parametric_arc(c, d)

    # Minimize distance between arcs
    return minimize(
        lambda t: np.arccos(np.dot(arc1(t[0]), arc2(t[1]))),
        [0.5, 0.5],
        bounds=([0, 1], [0, 1]),
    ).fun


# Get distances between Shapely Objects (in Degrees)
# But done properly in spherical coordinates
# Don't check above some minimum distance, as it is not worth it
def distance_between_shapely(sA, sB, min_dist=1):
    # Check if MultiPolygon (can only be sA)
    if hasattr(sA, 'geoms'):
        return np.min([distance_between_shapely(g, sB) for g in sA.geoms]).min()

    # Get SingleSphericalPolygons
    spA = SingleSphericalPolygon.from_radec(*sA.exterior.xy)
    spB = SingleSphericalPolygon.from_radec(*sB.exterior.xy)

    # Get points from spherical polygons
    spA_points = spA.points
    spB_points = spB.points

    # Don't bother checking if points are further apart than some threshhold (in degrees)
    if np.arccos(np.dot(spA_points[0], spB_points[0])) > np.deg2rad(min_dist):
        return np.inf

    # Get distances between arcs
    return np.rad2deg(
        np.min(
            [
                distance_between_arcs(*i)
                for i in [
                    (spA_points[i], spA_points[i + 1], spB_points[j], spB_points[j + 1])
                    for i in range(len(spA_points) - 1)
                    for j in range(len(spB_points) - 1)
                ]
            ]
        )
    )


if __name__ == '__main__':
    # Get list of proposal ids from the user
    parser = argparse.ArgumentParser(description='Compute Fields for JWST NIRISS')
    parser.add_argument(
        '--separation',
        type=float,
        help='Maximum separation between fields (in arcseconds)',
        default=30,
    )
    parser.add_argument(
        '--proposal_ids',
        type=str,
        nargs='+',
        help='List of proposal IDs to download',
        default=['1571', '3383', '4681'],
    )

    # Parse arguements
    args = parser.parse_args()
    separation = args.separation
    proposal_ids = args.proposal_ids

    # Query survey
    print('Querying MAST...')

    # Iterate over proposals and query mission interface
    results = vstack(
        [
            missions.query_criteria(
                program=pid,
                productLevel='1*',  # Require RAW assocation
                select_cols=[
                    'fileSetName',
                    'targ_ra',
                    'targ_dec',
                    'date_obs',
                    'program',
                    'effinttm',
                    'obs_id',
                ],
            )
            for pid in proposal_ids
        ]
    )

    # Get primary IDs for each observation
    results['primary_id'] = [int(o[1:6]) for o in results['obs_id']]

    # Rename columns
    del results['ArchiveFileID']
    results.rename_columns(
        [
            'obs_id',
            'fileSetName',
            'targ_ra',
            'targ_dec',
            'program',
            'effinttm',
        ],
        [
            'mission_id',
            'obs_id',
            'ra',
            'dec',
            'program_id',
            'exp_time',
        ],
    )
    results['obs_id'] = results['obs_id'] + '_nis'

    # Get all products
    obs = Observations.query_criteria(proposal_id=proposal_ids, obs_collection='JWST')

    # For each invidually extracted spectra, only keep one from each association (limit server load)
    isx = np.array([o.endswith('x1d.fits') for o in obs['dataURL']])
    xassoc = ['_'.join((s := o.split('_'))[0:1] + s[2:]) for o in obs[isx]['obs_id']]
    obs = vstack([obs[~isx], obs[isx][np.unique(xassoc, return_index=True)[1]]])

    # Get all products
    products = Observations.get_product_list(obs)
    products = Observations.filter_products(
        products, productSubGroupDescription='UNCAL'
    )
    products = products[np.unique(products['productFilename'], return_index=True)[1]]

    # Join products and observations (also restrict to columns we want)
    obs = obs['obsid', 's_region']
    obs.rename_column('obsid', 'parent_obsid')
    products = products[
        'obsID',
        'obs_id',
        'filters',
        'productFilename',
        'dataURI',
        'parent_obsid',
    ]
    products = join(products, obs, keys='parent_obsid')
    del products['parent_obsid']

    # Join with results
    obs = join(results, products, keys='obs_id')
    obs.sort('date_obs')

    # Compute regions
    print('Computing Overlapping Regions (Cartesian Approximation)...')
    regs = [SRegion(o).shapely[0] for o in obs['s_region']]
    obs = [Table(o) for o in obs]

    # Combine all overlapping fields
    fields = [obs[0]]
    f_regs = [regs[0]]
    obs = obs[1:]
    regs = regs[1:]

    # Iterate until all regions have been assigned to a field
    pbar = tqdm.tqdm(total=len(regs))
    while len(regs) > 0:
        # Iterate over regions
        for i, r in enumerate(regs):
            # Check if region intersects with most recent field (approximating in Cartesian)
            if f_regs[-1].intersects(r):
                # Combine regions and observations
                f_regs[-1] = f_regs[-1].union(regs.pop(i))
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

    # Combine fields that are too close
    print('Combine Nearly Fields (Spherical)...')

    # Check distance between fields (in Spherical)
    i = 0
    pbar = tqdm.tqdm(total=len(fields))
    while i < len(fields):
        # Check against all other fields
        for j in range(i + 1, len(fields)):
            # Check distance between regions
            if distance_between_shapely(f_regs[i], f_regs[j]) < separation / 3600:
                # Combine fields
                f_regs[i] = f_regs[i].union(f_regs.pop(j))
                fields[i] = vstack([fields[i], fields.pop(j)])

                # Break loop and start over
                i -= 1  # Decrement index to counteract increment
                break

        # Increment index
        i += 1
        pbar.update(1)
    pbar.close()

    # Get field names from Constellations
    names = []
    for i, f in enumerate(fields):
        # Create field Name
        con = get_constellation(
            SkyCoord(*f['ra', 'dec'][0], unit='deg'), short_name=True
        ).lower()

        # Find instance of constellation
        j = 0
        while f'{con}-{str(j).zfill(2)}' in names:
            j += 1
        name = f'{con}-{str(j).zfill(2)}'

        # Keep track
        names.append(name)

    # Write field names to file
    if not os.path.isdir('FIELDS'):
        os.mkdir('FIELDS')
    with open('FIELDS/fields.txt', 'w') as f:
        f.write('\n'.join(names))

    # Save fields to FITS file
    print('Saving to FITS...')
    obs_hdul = fits.HDUList([fits.PrimaryHDU()])
    prod_hdul = fits.HDUList([fits.PrimaryHDU()])
    for i, f in enumerate(fields):
        # Append field observations
        obs_hdul.append(fits.BinTableHDU(f))
        obs_hdul[-1].header['EXTNAME'] = names[i]

    # Save FITS file
    obs_hdul.writeto('FIELDS/fields.fits', overwrite=True)
