#! /usr/bin/env python

# Import packages
import os
import json
import glob
import shutil
import numpy as np
from tqdm import tqdm
from sregion import SRegion
from astropy.io import fits
from shapely import union_all
import xml.etree.ElementTree as et
from astropy.table import Table,vstack
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord,get_constellation

if __name__ == '__main__':

    # Set proposal IDs
    proposal_ids = ['1571','3383']

    # Query survey
    obs = Observations.query_criteria(
        proposal_id=proposal_ids,
        instrument_name="NIRISS*",
        obs_collection="JWST")
    obs.sort('t_obs_release') # Sort by release date

    # Download APT files
    if not os.path.isdir('APT'): os.mkdir('APT')
    for i in proposal_ids:
        os.system(f'wget -q --show-progress https://www.stsci.edu/jwst/phase2-public/{i}.aptx')
        os.system(f'bsdtar -xf {i}.aptx -C APT')
        os.system(f'rm {i}.aptx APT/manifest APT/edu.stsci.mpt.MsaPlanningToolMPT_UI_STATE.json')

    # Parse XML files
    xml = {i:et.parse(f'APT/{i}.xml').getroot() for i in proposal_ids}
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
            if found: break
    
    # Add primary IDs to table
    obs.add_column(pids,name='prim_id')

    # Compute regions
    regs = [SRegion(o).shapely[0] for o in obs['s_region']]
    obs = [Table(o) for o in obs]

    # Combine all overlapping clusters
    clusters = [obs[0]]
    c_regs = [regs[0]]
    obs = obs[1:]
    regs = regs[1:]

    # Iterate until all regions have been assigned to a cluster
    while len(regs) > 0:

        # Iterate over regions
        for i,r in enumerate(regs):

            # Check if region intersects with most recent cluster
            if c_regs[-1].intersects(r):

                # Combine regions and observations
                c_regs[-1] = c_regs[-1].union(regs.pop(i))
                clusters[-1] = vstack([clusters[-1],obs.pop(i)])

                # Break loop and start over
                notfinished = False
                break

            # Finished loop with no intersection
            notfinished = True

        # If no intersection, add new clusters
        if notfinished:
            c_regs.append(regs.pop(0))
            clusters.append(obs.pop(0))

    # Save clusters
    jmax = 0
    names = []
    for i,c in enumerate(clusters):

        # Create Cluster Name
        con = get_constellation(SkyCoord(*c['s_ra','s_dec'][0],unit='deg'),short_name=True).lower()

        # Find instance of constellation
        j = 0
        while f'{con}-{str(j).zfill(2)}' in names:
            j += 1
        name = f'{con}-{str(j).zfill(2)}'
        if j > jmax: jmax = j

        # Keep track
        names.append(name)


    # Save clusters to FITS file
    obs_cols = ['obs_id','s_ra','s_dec','filters','t_obs_release','t_exptime','obsid','s_region','prim_id'] # Columns to keep
    prod_cols = ['obs_id','productFilename','dataURI','obs_collection','obsID']
    obs_hdul = fits.HDUList([fits.PrimaryHDU()])
    prod_hdul = fits.HDUList([fits.PrimaryHDU()])
    for i,c in enumerate(tqdm(clusters)):

        # Append cluster observations
        obs_hdul.append(fits.BinTableHDU(c[obs_cols]))
        obs_hdul[-1].header['EXTNAME'] = names[i]

        # Query products (only science uncalibrated)
        allprods = Observations.get_product_list(c)
        good = np.logical_and(
            allprods['productType']=='SCIENCE',
            allprods['productSubGroupDescription']=='UNCAL'
            )
        prods = allprods[good]

        # Append products
        prod_hdul.append(fits.BinTableHDU(prods[prod_cols]))
        prod_hdul[-1].header['EXTNAME'] = names[i]

    # Save FITS file
    obs_hdul.writeto('cluster-obs.fits',overwrite=True)
    prod_hdul.writeto('cluster-prods.fits',overwrite=True)

    # Get products to download
    prods = vstack([Table(p.data) for p in prod_hdul[1:]])

    # Create Download Directory
    if not os.path.isdir('UNCAL'):
        os.mkdir('UNCAL')

    # Get list of files to download
    todo = np.setdiff1d(prods['productFilename'],[os.path.basename(f) for f in glob.glob('UNCAL/*.fits')])

    # Download products if not already downloaded
    if len(todo) > 0: Observations.download_products(join(prods,Table([todo])),download_dir='UNCAL',flat=True)
