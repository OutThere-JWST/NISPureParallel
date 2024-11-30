#! /usr/bin/env python

# Import packages
import os
import grizli.utils
from itertools import product
from multiprocessing import Pool


# Main function
def main():
    # Create directories
    directories = [
        grizli_dir := os.environ['GRIZLI'],
        conf := os.path.join(grizli_dir, 'CONF'),
        templates := os.path.join(grizli_dir, 'templates'),
        os.environ['iref'],
        os.environ['jref'],
        os.environ['CRDS_PATH'],
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Fetch config files
    grizli.utils.fetch_default_calibs(get_acs=False)
    grizli.utils.fetch_config_files(get_acs=False, get_jwst=True)
    if os.path.islink(templates):
        os.remove(templates)
        os.makedirs(templates)
    grizli.utils.symlink_templates(force=True)

    # Change to CONF directory
    os.chdir(conf)

    # Fetch updated NIRISS trace files
    # https://zenodo.org/records/7628094
    for filetype in ['config', 'sens']:
        filename = f'niriss_{filetype}_221215.tar.gz'
        url = f'https://zenodo.org/record/7628094/files/{filename}'
        if os.path.exists(filename):
            os.remove(filename)
        os.system(f'wget {url} -O {filename}')
        os.system(f'tar xzvf {filename}')

    # Multiprocess
    gfs = product(['CLEAR', 'GR150R', 'GR150C'], ['F115W', 'F150W', 'F200W'])
    with Pool(processes=9) as executor:
        executor.starmap(fetch_file, gfs)


# Fetch updated NIRISS WFSSBack files
# https://zenodo.org/records/13741413
def fetch_file(grism, filt):
    filename = f'nis-{filt}-{grism}_skyflat.fits'.lower()
    url = f'https://zenodo.org/record/13741413/files/{filename}'
    if os.path.exists(filename):
        os.remove(filename)
    os.system(f'wget {url} -O {filename}')


# Call main function
if __name__ == '__main__':
    main()
