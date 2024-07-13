#! /usr/bin/env python

# Import packages
import os
import shutil
import argparse
import numpy as np
from os import path
from glob import glob
from tqdm import trange, tqdm
from multiprocessing import cpu_count

# Image processing
from PIL import Image
from fitsmap import convert
from reproject import reproject_interp

# Astropy Packages
from astropy.io import fits
from astropy.table import Table, join

def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=(cpu_count() - 2))
    parser.add_argument('--slowsegmap', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    slowsegmap = args.slowsegmap
    fname = args.fieldname
    ncpu = args.ncpu

    # Print step
    print(f'Fitsmapping {fname}')

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    home = os.path.join(fields, fname)

    # Subdirectories
    prep = os.path.join(home, 'Prep')
    plots = os.path.join(home, 'Plots')
    extract = os.path.join(home, 'Extractions')
    fitsmap = path.join(home, 'fitsmap')

    # Remove last map
    if os.path.isdir(fitsmap):
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
    obs = Table.read(os.path.join(fields, 'field-obs.fits'), fname)
    filters = np.unique(obs['filters'])

    # Create image files (FITS)
    filts = [f.split(';')[1] for f in filters if 'CLEAR' in f]
    for f in filts:
        outfile = path.join(fitsmap, f'{f}.fits')
        shutil.copy(
            path.join(prep, f'{fname}-{f.lower()}n-clear_drc_sci.fits'), outfile
        )
        files.append(outfile)

    # Create grism files
    ref = fits.getheader(path.join(fitsmap, f'{f}.fits'))
    grisms = np.unique([f.split(';')[1] for f in filters if 'GR' in f])
    ngrisms = 0
    for g in grisms:
        # Get file pattern and PAs
        pre = f'{fname}-{g.lower()}-'
        pas = [
            f for f in glob(path.join(extract, f'{pre}*grism_sci*')) if 'proj' not in f
        ]

        # Iterate over PAs
        for pa in pas:
            ngrisms += 1

            # Get and project original image
            outproj = path.join(pa.replace('.fits', '_proj.fits'))
            if not path.exists(outproj):
                proj, _ = reproject_interp(pa, ref)
                fits.PrimaryHDU(proj).writeto(outproj)

            # Copy file to fitsmap
            outfile = path.join(
                fitsmap,
                f"{g}-{pa.split('/')[-1].split('_')[0].replace(f'{pre}','')}.fits",
            )
            shutil.copy(outproj, outfile)
            files.append(outfile)

    # Create Segmap (if it doesn't exist)
    print('Creating Segmentation Image')
    palette = [
        [226, 1, 52],
        [0, 141, 249],
        [0, 159, 129],
        [255, 195, 59],
        [255, 90, 175],
        [159, 1, 98],
    ]
    if not path.exists(path.join(plots, 'Segmentation.png')):
        # Either slow-segmap coloring with 4 Color Theorem (Fast Version)
        # Or fast-segmap coloring with simple coloring
        if slowsegmap:
            # Load Segmap
            seg = fits.getdata(path.join(prep, f'{fname}-ir_seg.fits'))[
                ::-1
            ]  # Flip for orientation

            # Four color theorem
            verts = {}
            for b in trange(1, seg.max() + 1):
                # Create list of edges
                verts[b] = []

                # Iterate over all pixels in bin
                locs = np.where(seg == b)
                for x, y in zip(*locs):
                    # Iterate over all touching edges
                    for i in [-1, 1]:
                        # Horizontal touches
                        h = seg[x + i, y]
                        if (h != b) and (h != 0) and (h not in verts[b]):
                            verts[b].append(h)

                        # Vertical Touches
                        v = seg[x, y + i]
                        if (v != b) and (v != 0) and (v not in verts[b]):
                            verts[b].append(v)

            # Color Vertices
            vertices = sorted((list(verts.keys())))
            color_graph = {}
            for vertex in tqdm(vertices):
                unused_colors = len(vertices) * [True]
                for neighbor in verts[vertex]:
                    if neighbor in color_graph:
                        color = color_graph[neighbor]
                        unused_colors[color] = False
                for color, unused in enumerate(unused_colors):
                    if unused:
                        color_graph[vertex] = color
                        break

            # Create segmap image
            im = np.zeros(seg.shape + (3,), dtype='uint8')
            for c in tqdm(color_graph):
                im[seg == c] = palette[color_graph[c]]

        else:
            # Easy Color
            im = np.zeros(seg.shape + (3,), dtype='uint8')
            for i, p in enumerate(palette):
                im[seg % len(palette) == i + 1] = p

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
            imgs = [i for i in imgs if os.path.exists(os.path.join(extract, i))]
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
    ffiles = [os.path.join(fitsmap, f) for f in files]
    convert.files_to_map(
        ffiles,
        out_dir=path.join(fitsmap, fname),
        norm_kwargs=norm_kwargs,
        cat_wcs_fits_file=path.join(fitsmap, f'{filts[0]}.fits'),
        task_procs=len(ffiles),
        procs_per_task=ncpu // (len(ffiles) + 1),
    )

    # Copy spectra images over
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
                    continue  # Ignoreif we don't have to do anything

                # Get keypair
                keypair = m.split(':')

                # Get label
                label = keypair[0].split('_')

                # Fix label
                keypair[0] = f'{label[0]} ({label[1]}.{label[2][:-1]})"'

                # Join and replace\
                menuitems[j] = ':'.join(keypair)

            # Replace
            js[i + 1] = ','.join(menuitems)

            break

if __name__ == '__main__':
    main()