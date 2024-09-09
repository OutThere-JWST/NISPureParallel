#! /usr/bin/env python

# Python Packages
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

# Computational Packages
import numpy as np

# Astropy Packages
from astropy.table import Table, join
from astroquery.mast import Observations

if __name__ == '__main__':
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Get paths
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    prods = Table.read(os.path.join(fields, 'field-prods.fits'), fname)
    home = os.path.join(fields, fname)
    uncal = os.path.join(home, 'UNCAL')

    # Destination directory
    home = os.path.join(fields, fname)

    # Create directory tree if it doesn't exist
    if not os.path.exists(home):
        os.makedirs(home)

    print(f'Downloading Products for {fname}...')

    # Get list of files to download
    todo = np.setdiff1d(prods['productFilename'], os.listdir(uncal))

    # Download products if not already downloaded
    if len(todo) > 0:
        # Products to download
        todo_prods = join(prods, Table([todo]))

        # Multi-threaded download
        if ncpu > 1:
            with ThreadPoolExecutor(ncpu) as executor:
                executor.map(
                    lambda p: Observations.download_products(
                        p, download_dir=uncal, flat=True
                    ),
                    todo_prods,
                )

        # Single-threaded download
        else:
            Observations.download_products(todo_prods, download_dir=uncal, flat=True)
    
    print(f'Downloaded Products for {fname}')
