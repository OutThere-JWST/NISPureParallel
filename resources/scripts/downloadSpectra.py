#! /usr/bin/env python

# Import packages
import os
import argparse
import subprocess
import numpy as np
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor


# Download product with rsync
def download_spectrum(extract, local_dir, remote):
    """Download product from remote server."""

    # Product name
    field = extract['root']

    # Remote URL
    remote_url = os.path.join(remote, field, 'spectra')

    # Files
    types = ['1D', 'beams', 'full', 'row', 'stack']
    files = [f'{field}_{str(extract['NUMBER']).zfill(5)}_{t}.fits' for t in types]

    # Execute command
    for file in files:
        # Download command
        command = [
            'rsync',
            '-avz',
            os.path.join(remote_url, file),
            os.path.join(local_dir, field, 'spectra'),
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
        '--remote', type=str, help='Remote URL', default='outthere-mpia.org/s3/data'
    )
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()

    # Load extracted
    extracted = Table.read(args.extracted)

    # Remote URL
    remote = args.remote

    # Number of CPUs
    ncpu = args.ncpu

    # Create directories
    home = os.getcwd()
    for field in np.unique(extracted['root']):
        os.makedirs(os.path.join(home, field, 'spectra'), exist_ok=True)

    # Multi-threaded download
    if ncpu > 1:
        with ThreadPoolExecutor(ncpu) as executor:
            executor.map(
                lambda e: download_spectrum(e, remote),
                extracted,
            )

    # Single-threaded download
    else:
        for extract in extracted:
            download_spectrum(extract, home, remote)


if __name__ == '__main__':
    main()
