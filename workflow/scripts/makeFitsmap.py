#! /usr/bin/env python

import argparse
import os
import shutil
from glob import glob
from os import path

import networkx as nx
import numpy as np
from astropy.io import fits
from astropy.table import Table, join
from fitsmap import convert
from PIL import Image
from tqdm import trange


def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    fname = args.fieldname
    ncpu = args.ncpu

    # Print step
    print(f'Fitsmapping {fname}')

    # Get paths and get fields
    main = os.getcwd()
    fields = path.join(main, 'FIELDS')
    home = path.join(fields, fname)

    # Subdirectories
    prep = path.join(home, 'Prep')
    plots = path.join(home, 'Plots')
    extract = path.join(home, 'Extractions')
    fitsmap = path.join(home, 'fitsmap')

    # Remove last map
    if path.isdir(fitsmap):
        shutil.rmtree(fitsmap)
    os.mkdir(fitsmap)

    # Keep track of files for fitsmap creation
    files = []

    # Create RGB file
    shutil.copy(path.join(plots, f'{fname}.rgb.png'), path.join(fitsmap, 'RGB.png'))
    files.append(path.join(fitsmap, 'RGB.png'))
    for f in glob(path.join(plots, f'{fname}-grism.*_sci.png')):
        split = path.basename(f).split('.')
        pa = '.'.join([split[1], split[2].split('_')[0]])
        shutil.copy(f, path.join(fitsmap, f'RGB-{pa}.png'))
        files.append(path.join(fitsmap, f'RGB-{pa}.png'))

    # Get Observations and filters
    obs = Table.read(path.join(fields, 'fields.fits'), fname)

    # Create image files (FITS)
    filts = np.unique(obs['nis_pupil'][obs['nis_pupil'] == 'CLEAR'])
    for f in filts:
        # Direct image
        outfile = path.join(fitsmap, f'{f}.fits')
        shutil.copy(
            path.join(prep, f'{fname}-{f.lower()}n-clear_drc_sci.fits'), outfile
        )
        files.append(outfile)

        # Exposure time map
        outfile = path.join(fitsmap, f'{f}_exp.fits')
        shutil.copy(
            path.join(prep, f'{fname}-{f.lower()}n-clear_drc_exp.fits'), outfile
        )
        files.append(outfile)

    # Create grism files
    grisms = np.unique(obs['nis_pupil'][obs['nis_pupil'] != 'CLEAR'])
    for g in grisms:
        # Get file pattern and PAs
        pre = f'{fname}-{g.lower()}-'
        pas = [
            path.basename(f).split('_')[0].replace(f'{pre}', '')
            for f in glob(path.join(extract, f'{pre}*grism_sci.fits'))
            if 'proj' not in f
        ]

        # Iterate over PAs
        for pa in pas:
            # Copy projection to fitsmap
            proj = path.join(extract, f'{pre}{pa}_grism_sci_proj.fits')
            outfile = path.join(fitsmap, f'{g}-{pa}.fits')
            shutil.copy(proj, outfile)
            files.append(outfile)

            # Save exposure time map
            exp = path.join(extract, f'{pre}{pa}_grism_exp.fits')
            outfile = path.join(fitsmap, f'{g}-{pa}_exp.fits')
            shutil.copy(exp, outfile)
            files.append(outfile)

    # Create Segmap
    print('Creating Segmentation Image')

    # Get segmentation array
    seg = fits.getdata(path.join(prep, f'{fname}-ir_seg.fits'))[::-1]

    # Define palette
    palette = [
        [226, 1, 52],  # Alizarin Crimson
        [0, 141, 249],  # Dodger Blue
        [0, 159, 129],  # Jeepers Creepers
        [255, 195, 59],  # Bright Spark
        [255, 90, 175],  # Barbie Pink
        [159, 1, 98],  # Jazzberry Jam
        [255, 178, 253],  # Plum
        [132, 0, 205],  # French Violet
        [0, 252, 207],  # Aquamarine
        [164, 1, 34],  # Carmine
    ]

    # Color the segmentation array
    color_map = greedy_coloring_ordered(seg, len(palette))

    # Map the segmentation to the palette index using color_graph
    mapped_indices = np.vectorize(color_map.get)(seg)

    # Create segmap image
    im = np.zeros(seg.shape + (3,), dtype='uint8')

    # Apply the palette to the image
    for i in range(mapped_indices.max()):
        im[mapped_indices == i + 1] = palette[i]

    # Save figure
    out = Image.fromarray(im, 'RGB')
    out.save(path.join(plots, 'Segmentation.png'))

    # Copy over
    files.append(path.join(plots, 'Segmentation.png'))

    # Detection Catalog
    cat = Table.read(path.join(prep, f'{fname}-ir.cat.fits'))
    columns = ['NUMBER', 'RA', 'DEC', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'MAG_AUTO']
    detection = cat[columns]
    detection.rename_columns(columns, ['id', 'ra', 'dec', 'a', 'b', 'theta', 'mag'])
    detection['theta'] *= 180 / np.pi  # Convert to degrees
    detection.write(path.join(fitsmap, 'Detection.cat'), format='ascii.csv')
    files.append(path.join(fitsmap, 'Detection.cat'))

    # Create fitting catalog
    results_file = path.join(extract, f'{fname}_fitresults.fits')
    if path.exists(results_file):
        results = Table.read(results_file)
        results = results['id', 'redshift']

        # path.join catalogs
        extraction = join(detection, results)['id', 'ra', 'dec', 'redshift']

        # Create image links
        links, allims = [], []
        for i in results['id']:
            # Iterate over image suffixes
            imgs = [
                f'{fname}_{str(i).zfill(5)}.{suffix}.png'
                for suffix in ['stack', 'line', 'full']
            ]
            imgs = [i for i in imgs if path.exists(path.join(extract, i))]
            allims.extend(imgs)

            # Append to list of extracted links
            links.append(
                '<br>'.join([f'<img src="spectra/{f}" width="500">' for f in imgs])
            )
        extraction['extraction'] = links

        # Write Catalog
        extraction.write(path.join(fitsmap, 'Extraction.cat'), format='ascii.csv')
        files.append(path.join(fitsmap, 'Extraction.cat'))

    # Create map
    norm_kwargs = dict(stretch='log', min_percent=30, max_percent=99.9)
    ffiles = [path.join(fitsmap, f) for f in files]
    convert.files_to_map(
        ffiles,
        title=f'{fname} FitsMap',
        out_dir=path.join(fitsmap, fname),
        norm_kwargs=norm_kwargs,
        cat_wcs_fits_file=path.join(fitsmap, f'{filts[0]}.fits'),
        task_procs=len(ffiles),
        procs_per_task=max(ncpu // (len(ffiles) + 1), 1),
        units_are_pixels=False,
        pixel_scale=0.04,  # Same as in mosaic/contam
    )

    # Copy spectra images over
    if path.exists(results_file):
        os.mkdir(path.join(fitsmap, fname, 'spectra'))
        for f in allims:
            shutil.copy(path.join(extract, f), path.join(fitsmap, fname, 'spectra'))

    # FitsMap Add Ons
    with open(path.join(fitsmap, fname, 'js/index.js'), 'r') as f:
        js = f.readlines()

    # Fix menu labels
    for i, line in enumerate(js):
        # Find correct line
        if 'layerControl' in line:
            line = js[i + 1]

            # Get all menuitems
            menuitems = line.split(',')

            # Iterate over menuitems
            for j, m in enumerate(menuitems):
                if '_' not in m:
                    continue  # Ignore if we don't have to do anything

                # Get keypair
                keypair = m.split(':')

                # Get label
                label = keypair[0].split('_')

                # Fix label
                keypair[0] = f'{label[0]} ({".".join(label[1:])})'.replace(
                    '")', ')"'
                ).replace('.exp', ' exp')

                # Join and replace\
                menuitems[j] = ':'.join(keypair)

            # Replace
            js[i + 1] = ','.join(menuitems)

            break

    # Save
    with open(path.join(fitsmap, fname, 'js/index.js'), 'w+') as f:
        f.write(''.join(js))


# Define function to convert array to graph
def greedy_coloring_ordered(array, max_colors):
    # Get the shape of the array
    rows, cols = array.shape

    # Create an empty graph
    graph = nx.Graph()

    # Iterate over the array
    for r in trange(rows):
        for c in range(cols):
            # Get the current value
            current_value = array[r, c]

            # Check the right neighbor
            if c + 1 < cols:
                right_value = array[r, c + 1]

                # If the values are different, add an edge
                if current_value != right_value:
                    graph.add_edge(current_value, right_value)

            # Check the bottom neighbor
            if r + 1 < rows:
                bottom_value = array[r + 1, c]

                # If the values are different, add an edge
                if current_value != bottom_value:
                    graph.add_edge(current_value, bottom_value)

    # Sort the nodes by degree
    color_map = {}
    nodes_sorted_by_degree = sorted(
        graph.nodes(), key=lambda x: graph.degree(x), reverse=True
    )

    # Greedy coloring algorithm
    for node in nodes_sorted_by_degree:
        # Get the colors of the neighbors
        neighbor_colors = {
            color_map[neighbor]
            for neighbor in graph.neighbors(node)
            if neighbor in color_map
        }

        # Assign the first available color
        for color in range(max_colors):
            if color not in neighbor_colors:
                color_map[node] = color
                break
    return color_map


if __name__ == '__main__':
    main()
