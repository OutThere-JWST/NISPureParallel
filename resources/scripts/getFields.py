#! /usr/bin/env python

# Python Packages
import os
import glob
import shutil
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as et

# Computational Packages
import numpy as np
from scipy.optimize import minimize

# Astropy Packages
from astropy.io import fits
from astroquery.mast import Observations
from astropy.table import Table, join, vstack
from astropy.coordinates import SkyCoord, get_constellation

# Geometry packages
from sregion import SRegion
from spherical_geometry.polygon import SingleSphericalPolygon


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
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()

    # Set proposal IDs
    proposal_ids = ['1571', '3383', '4681']

    # Query survey
    print('Querying MAST...')
    obs = Observations.query_criteria(
        proposal_id=proposal_ids, instrument_name='NIRISS*', obs_collection='JWST'
    )
    obs.sort('t_obs_release')  # Sort by release date

    # Download APT files
    print('Downloading APT files...')
    if not os.path.isdir('APT'):
        os.mkdir('APT')
    for i in proposal_ids:
        os.system(
            f'wget -q --show-progress https://www.stsci.edu/jwst/phase2-public/{i}.aptx'
        )
        os.system(f'bsdtar -xf {i}.aptx -C APT')
        os.system(
            f'rm {i}.aptx APT/manifest APT/edu.stsci.mpt.MsaPlanningToolMPT_UI_STATE.json'
        )

    # Parse XML files
    xml = {i: et.parse(f'APT/{i}.xml').getroot() for i in proposal_ids}
    shutil.rmtree('APT')

    # Associate with primary observation
    ns = '{http://www.stsci.edu/JWST/APT}'
    pids = []
    for o in obs:
        # Get ObsID
        oid = o['obs_id'][7:10]
        found = False

        # Parse XML file
        dr = xml[o['proposal_id']].find(f'{ns}DataRequests')
        for og in dr.findall(f'{ns}ObservationGroup'):
            for ob in og.findall(f'{ns}Observation'):
                # Get Observation number
                if oid == ob.find(f'{ns}Number').text.zfill(3):
                    # Get primary ID
                    pids.append(ob.find(f'{ns}PureParallelSlotGroupName').text[:4])
                    found = True
                    break

            # Break if found
            if found:
                break

    # Add primary IDs to table
    obs.add_column(pids, name='prim_id')

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
    while len(regs) > 0:
        # Iterate over regions
        for i, r in enumerate(regs):
            # Check if region intersects with most recent field (approximating in Cartesian)
            if f_regs[-1].intersects(r):
                # Combine regions and observations
                f_regs[-1] = f_regs[-1].union(regs.pop(i))
                fields[-1] = vstack([fields[-1], obs.pop(i)])

                # Break loop and start over
                notfinished = False
                break

            # Finished loop with no intersection
            notfinished = True

        # If no intersection, add new fields
        if notfinished:
            f_regs.append(regs.pop(0))
            fields.append(obs.pop(0))

    # Combine fields that are too close
    print('Combine Nearly Fields (Spherical)...')

    # Check distance between fields (in Spherical)
    i = 0
    while i < len(fields):
        # Check against all other fields
        for j in range(i + 1, len(fields)):
            # Check distance between regions
            if distance_between_shapely(f_regs[i], f_regs[j]) < 30 / 3600:
                # Combine fields
                f_regs[i] = f_regs[i].union(f_regs.pop(j))
                fields[i] = vstack([fields[i], fields.pop(j)])

                # Break loop and start over
                i -= 1  # Decrement index to counteract increment
                break

        # Increment index
        i += 1

    # Get field names from Constellations
    names = []
    for i, f in enumerate(fields):
        # Create field Name
        con = get_constellation(
            SkyCoord(*f['s_ra', 's_dec'][0], unit='deg'), short_name=True
        ).lower()

        # Find instance of constellation
        j = 0
        while f'{con}-{str(j).zfill(2)}' in names:
            j += 1
        name = f'{con}-{str(j).zfill(2)}'

        # Keep track
        names.append(name)

    # Write field names to file
    i = names.index('leo-08')
    names.remove('leo-08')
    fields.pop(i)
    print('Removing leo-08 while it is bugged')
    if not os.path.isdir('FIELDS'):
        os.mkdir('FIELDS')
    with open('FIELDS/fields.txt', 'w') as f:
        f.write('\n'.join(names))

    # Save fields to FITS file
    print('Querying Products and Saving to FITS...')
    obs_cols = [
        'obs_id',
        's_ra',
        's_dec',
        'filters',
        't_obs_release',
        't_exptime',
        'obsid',
        's_region',
        'prim_id',
    ]  # Columns to keep
    prod_cols = ['obs_id', 'productFilename', 'dataURI', 'obs_collection', 'obsID']
    obs_hdul = fits.HDUList([fits.PrimaryHDU()])
    prod_hdul = fits.HDUList([fits.PrimaryHDU()])
    for i, f in enumerate(tqdm(fields)):
        # Append field observations
        obs_hdul.append(fits.BinTableHDU(f[obs_cols]))
        obs_hdul[-1].header['EXTNAME'] = names[i]

        # Query products (only science uncalibrated)
        allprods = Observations.get_product_list(f)
        good = np.logical_and(
            allprods['productType'] == 'SCIENCE',
            allprods['productSubGroupDescription'] == 'UNCAL',  # Uncal files only
        )
        prods = allprods[good]

        # Append products
        prod_hdul.append(fits.BinTableHDU(prods[prod_cols]))
        prod_hdul[-1].header['EXTNAME'] = names[i]

    # Save FITS file
    obs_hdul.writeto('FIELDS/field-obs.fits', overwrite=True)
    prod_hdul.writeto('FIELDS/field-prods.fits', overwrite=True)

    # Skip if download flag is not set
    if args.download:
        print('Downloading Products...')

        # Get products to download
        prods = vstack([Table(p.data) for p in prod_hdul[1:]])

        # Create Download Directory
        if not os.path.isdir('UNCAL'):
            os.mkdir('UNCAL')

        # Get list of files to download
        todo = np.setdiff1d(
            prods['productFilename'],
            [os.path.basename(f) for f in glob.glob('UNCAL/*.fits')],
        )

        # Download products if not already downloaded
        if len(todo) > 0:
            Observations.download_products(
                join(prods, Table([todo])), download_dir='UNCAL', flat=True
            )