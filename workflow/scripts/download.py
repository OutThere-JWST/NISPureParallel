#! /usr/bin/env python

# Python Packages
import os
import argparse

# Astropy Packages
from astroquery.mast import MastMissions

if __name__ == '__main__':
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    args = parser.parse_args()
    file = args.file

    MastMissions(mission='jwst').download_file(os.path.basename(file), local_path=file)
