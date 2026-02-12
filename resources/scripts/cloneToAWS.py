#! /usr/bin/env python

import glob
import os
import subprocess
from os import path

import toml
from astropy.io import fits
from astropy.table import Table, vstack

# Remote Path
remote_path = 'nppS3:nirisspureparallel'

# Default Options
N, B = 16, 64
default_options = [
    f'--transfers={N}',
    f'--checkers={N}',
    f'--buffer-size={B}M',
    f'--s3-upload-concurrency={N}',
    '-v',
]


# Function to copy files
def copy_files(filelist, local_dir, remote_dir, options=default_options):
    """Function to rclone a list of files to the remote server."""

    # Relative paths
    relfiles = [path.relpath(f, local_dir) for f in filelist]

    # Create a file with the list of files
    file = 'files.txt'
    with open(file, 'w') as f:
        for item in relfiles:
            f.write(f'{item}\n')

    # Copy command
    command = ['rclone', 'copy', f'--files-from={file}', local_dir, remote_dir]

    # Execute Command
    try:
        subprocess.run(command + options, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Failed to copy to {remote_dir}. Error: {e}')

    # Remove the file
    os.remove(file)

    # Return the list of files
    return relfiles


# Function to sync directory
def sync_dir(local_dir, remote_dir, options=default_options):
    """Function to rclone a list of files to the remote server."""

    # Copy command
    command = ['rclone', 'sync', local_dir, remote_dir]

    # Execute Command
    try:
        subprocess.run(command + options, check=True)
        print(' '.join(command + options))
    except subprocess.CalledProcessError as e:
        print(f'Failed to copy to {remote_dir}. Error: {e}')


def main():
    # Local path to FIELDS directory
    local_path = path.join(os.getcwd(), 'FIELDS')

    # Create remote directory
    data_remote_path = path.join(remote_path, 'data')

    # Get list of fields
    fields = [f.name for f in fits.open(path.join(local_path, 'fields.fits'))[1:]]

    # Iterate over fields
    phot_combined = []
    zfit_combined = []
    manifest = {}
    for field in fields:
        print('Cloning', field)

        # Field Path
        field_path = path.join(local_path, field)

        # Fitsmap Sync
        fitsmap_path = path.join(field_path, 'fitsmap', field)
        fitsmap_remote_path = path.join(remote_path, 'maps', field)
        sync_dir(fitsmap_path, fitsmap_remote_path)

        # Prep Catalog Paths
        prep_path = path.join(field_path, 'Prep')
        catstrings = [
            f'{field}-ir*.fits',
            f'{field}-ir.reg',
            f'{field}-f*n-clear*fits',
            f'{field}-f*n-clear_wcs.csv',
        ]
        catalogs = []
        map(catalogs.extend, [glob.glob(path.join(prep_path, s)) for s in catstrings])

        # Copy Over Catalogs
        catalogs_remote_path = path.join(data_remote_path, field)
        files = copy_files(catalogs, prep_path, catalogs_remote_path)

        # Extraction Catalogs
        extract_path = path.join(field_path, 'Extractions')
        catalogs = glob.glob(path.join(extract_path, f'{field}-f*_grism_*.fits'))
        catalogs = [c for c in catalogs if 'proj' not in c]  # Remove proj
        catalogs += glob.glob(path.join(extract_path, f'{field}-extracted.*'))

        # Handle Photometric and Fit Results
        phot = path.join(extract_path, f'{field}_phot_apcorr.fits')
        if path.exists(phot):
            catalogs.append(phot)
            phot = Table.read(phot)
            phot.add_column(field, name='root', index=0)
            phot_combined.append(phot)
        zfit = path.join(extract_path, f'{field}_fitresults.fits')
        if path.exists(zfit):
            catalogs.append(zfit)
            zfit_combined.append(Table.read(zfit))

        # Copy Over Catalogs
        files += copy_files(catalogs, extract_path, catalogs_remote_path)

        # Create Manifest
        manifest[field] = {
            'direct': [f for f in files if 'clear' in f],
            'grism': [f for f in files if 'grism' in f],
            'detection': [f for f in files if '-ir_' in f],
            'summary': [
                f
                for f in [
                    f'{field}-ir.cat.fits',
                    f'{field}-extracted.fits',
                    f'{field}_fitresults.fits',
                    f'{field}_phot_apcorr.fits',
                ]
                if f in files
            ],
            'regions': [f for f in files if f.endswith('.reg')],
        }

        # Save and copy over manifest
        manifest_name = f'MANIFEST-{field}.toml'
        with open(path.join(field_path, manifest_name), 'w') as f:
            toml.dump(manifest[field], f)
        copy_files(
            [path.join(field_path, manifest_name)], field_path, catalogs_remote_path
        )

        # Spectra Paths
        extensions = ['1D', 'beams', 'full', 'stack', 'row']
        spectra = sum(
            [glob.glob(path.join(extract_path, f'*{ext}.fits')) for ext in extensions],
            [],
        )

        # Copy Over Spectra
        spectra_remote_path = path.join(catalogs_remote_path, 'spectra')
        copy_files(spectra, extract_path, spectra_remote_path)

    # Create total catalogs
    names = ['phomoetry.fits', 'spectra-fitting.fits']
    for cats, name in zip([phot_combined, zfit_combined], names):
        # Stack and save
        vstack(cats, metadata_conflicts='silent').write(name)

    # Save Manifest
    manifest_name = 'MANIFEST.toml'
    with open(manifest_name, 'w') as f:
        toml.dump(manifest, f)
    names += [manifest_name]

    # Copy over the total catalogs
    copy_files(names, os.getcwd(), data_remote_path)

    # Delete the individual catalogs
    for name in names:
        os.remove(name)


if __name__ == '__main__':
    main()
