#! /usr/bin/env python

# Import packages
import os
import glob
import shutil
import argparse
import warnings
import numpy as np
from matplotlib import pyplot

# Astropy packages
from astropy.wcs import WCS
from astropy.table import Table

# grizli packages
import grizli
from grizli import utils
from grizli.aws import visit_processor
from grizli.pipeline import auto_script

# Silence warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    cname = args.fieldname

    # Print version and step
    print(f'Mosaicing {cname}')
    print(f'grizli:{grizli.__version__}')

    # Get paths and get fields
    main = os.getcwd()
    fields = os.path.join(main, 'FIELDS')
    obs = Table.read(os.path.join(fields, 'field-obs.fits'), cname)
    home = os.path.join(fields, cname)

    # Subdirectories
    prep = os.path.join(home, 'Prep')
    plots = os.path.join(home, 'Plots')

    # Go to prep directory
    os.chdir(prep)

    # Make a table with file information
    files = sorted(glob.glob('*rate.fits'))
    res = visit_processor.res_query_from_local(files=files)
    is_grism = np.array(['GR' in filt for filt in res['filter']])

    # Mosaic WCS
    hdu = utils.make_maximal_wcs(
        files=files, pixel_scale=0.04, pad=4, get_hdu=True, verbose=False
    )
    ref_wcs = WCS(hdu.header)

    # Drizzle full mosaics
    visit_processor.cutout_mosaic(
        cname,
        res=res[~is_grism],  # Just direct
        ir_wcs=ref_wcs,
        half_optical=False,  # Otherwise will make JWST exposures at half pixel scale of ref_wcs
        kernel='square',  # Drizzle parameters
        pixfrac=0.8,
        clean_flt=False,  # Otherwise removes "rate.fits" files from the working directory!
        s3output=None,
        make_exptime_map=False,
        weight_type='jwst',
        skip_existing=False,
    )

    # Get filters for RGB
    scales = {'f200wn-clear': 1.65, 'f150wn-clear': 1.33, 'f115wn-clear': 1.0}
    filts = np.sort(
        [
            f.split(';')[1].lower() + 'n-clear'
            for f in np.unique(obs['filters'])
            if 'CLEAR' in f
        ]
    )[::-1]
    scale = [scales[k] if k in filts else 0 for k in scales.keys()]

    # Create missing RGB filters
    todelete = []
    for k in scales.keys():
        if k not in filts:
            # Copy file
            shutil.copyfile(
                f'{cname}-{filts[0]}_drc_sci.fits', f'{cname}-{k}_drc_sci.fits'
            )
            todelete.append(f'{cname}-{k}_drc_sci.fits')

    # Create RGB Figure
    _, _, _, fig = auto_script.field_rgb(
        root=cname,
        HOME_PATH=None,
        force_rgb=scales.keys(),
        suffix='.rgb',
        show_ir=False,
        rgb_scl=scale,
        use_imsave=False,
        full_dimensions=1,
        get_rgb_array=False,
        scl=8,
        output_format='png',
        add_labels=False,
    )
    ax = fig.axes[0]

    # Add labels
    for i, k in enumerate(scales.keys()):
        f = k.replace('n-clear', '')
        ax.text(
            0.03 + 0.1 * i,
            0.97,
            f.upper(),
            color='rgb'[i],
            bbox=dict(facecolor='w', alpha=1),
            size=14,
            ha='left',
            va='top',
            transform=ax.transAxes,
        )

    # Save figure
    fig.tight_layout(pad=0)
    fig.savefig(f'{cname}.rgb.png', pad_inches=0, bbox_inches='tight')
    pyplot.close(fig)
    os.rename(
        os.path.join(prep, f'{cname}.rgb.png'), os.path.join(plots, f'{cname}.rgb.png')
    )

    # Delete extras
    for f in todelete:
        os.remove(f)

    # Create combined catalog
    auto_script.make_filter_combinations(
        cname,
        weight_fnu=True,
        min_count=1,
        filter_combinations={'ir': ['F115WN-CLEAR', 'F150WN-CLEAR', 'F200WN-CLEAR']},
    )
    # grizli.prep.make_SEP_catalog(f'{root}-ir', threshold=1.2)
    phot = auto_script.multiband_catalog(
        field_root=cname,
        detection_filter='ir',
        get_all_filters=True,
        rescale_weight=True,
    )
