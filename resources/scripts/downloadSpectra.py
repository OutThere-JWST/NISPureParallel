#! /usr/bin/env python

import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.table import Table


# Download product with rsync
def download_spectrum(extract, local_dir, remote, password):
    """Download product from remote server."""

    # Product name
    field = extract['field']

    # Remote URL
    remote_url = os.path.join(remote, field, 'spectra')

    # Files
    types = ['1D', 'beams', 'full', 'row', 'stack']
    files = [f'{field}_{str(extract["NUMBER"]).zfill(5)}.{t}.fits' for t in types]

    # Execute command
    for file in files:
        # Download command
        command = [
            'curl',
            '-u',
            f'outthere:{password}',
            '-o',
            os.path.join(local_dir, field, 'spectra'),
            os.path.join(remote_url, file),
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Failed to download {file}. Error: {e}')


# Main Function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('extracted', type=str, help='Path to extracted table')
    parser.add_argument(
        '--remote',
        type=str,
        help='Remote URL',
        default='https://outthere-mpia.org/s3/data',
    )
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # Prompt User for input
    print('Enter the password to the remote server')
    password = input()

    # Load extracted
    extracted = Table.read(args.extracted)

    # Remote URL
    remote = args.remote

    # Number of CPUs
    ncpu = args.ncpu

    # Create directories
    home = os.getcwd()
    for field in np.unique(extracted['field']):
        os.makedirs(os.path.join(home, field, 'spectra'), exist_ok=True)

    # Multi-threaded download
    if ncpu > 1:
        with ThreadPoolExecutor(ncpu) as executor:
            executor.map(
                lambda e: download_spectrum(e, home, remote, password), extracted
            )

    # Single-threaded download
    else:
        for extract in extracted:
            download_spectrum(extract, home, remote, password)


if __name__ == '__main__':
    main()
