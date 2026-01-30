#! /usr/bin/env python

import os
import argparse
import subprocess

import toml


# Download products with rsync
def download_product(product, prod_dir, remote, password):
    """Download product from remote server."""

    # Download command
    command = [
        'curl',
        '-u',
        f'outthere:{password}',
        '-o',
        os.path.join(prod_dir, product),
        os.path.join(remote, product),
    ]

    # Execute command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Failed to download {product}. Error: {e}')


# Main function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=str, help='Path to manifest file')
    parser.add_argument(
        '--remote',
        type=str,
        help='Remote URL',
        default='https://outthere-mpia.org/s3/data',
    )
    args = parser.parse_args()

    # Prompt User for input
    print('Enter the password to the remote server')
    password = input()

    # Load manifest
    manifest = toml.load(args.manifest)

    # Remote URL
    remote = args.remote

    # Loop over fields
    for field in manifest.keys():
        # Make field directory
        field_dir = os.path.join(os.getcwd(), field)
        os.makedirs(field_dir, exist_ok=True)

        # Get field product types
        field_prod_types = manifest[field]

        # Loop over product directories
        for prod_type in field_prod_types:
            # Make product directory
            prod_dir = os.path.join(field_dir, prod_type)
            os.makedirs(prod_dir, exist_ok=True)

            # Get list of products
            products = field_prod_types[prod_type]

            # Loop over products
            for product in products:
                # Download product
                download_product(
                    product, prod_dir, os.path.join(remote, field), password
                )


if __name__ == '__main__':
    main()
