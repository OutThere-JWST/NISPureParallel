#! /usr/bin/env python

# Import packages
import os
import argparse

# Multiprocessing
from threadpoolctl import threadpool_limits

# Astropy
from astropy.io import fits

# JWST Pipeline
import jwst
from columnjump import ColumnJumpStep
from jwst.pipeline import Detector1Pipeline, Image2Pipeline
from jwst.clean_flicker_noise import clean_flicker_noise as cfn


# Run pipeline in parallel
def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('uncal', type=str)
    parser.add_argument('rate', type=str)
    parser.add_argument('--scratch', action='store_true')
    args = parser.parse_args()
    uncal = args.uncal
    rate = args.rate

    # If exists and not forcing from scratch, skip
    if os.path.exists(rate) and not args.scratch:
        print(f'{rate} exists, skipping')
        return

    # Run pipeline
    cal(uncal, rate)


# Detector 1 Pipeline
@threadpool_limits.wrap(limits=1, user_api='blas')
def cal(file, out):
    print(f'Processing {file} with jwst: {jwst.__version__}')

    # Set ColumnJumpStep Parameters
    cjs = ColumnJumpStep()
    cjs.nsigma1jump, cjs.nsigma2jumps = 5.0, 5.0

    # Set 1/f parameters
    imaging = (
        fits.getval(file, 'FILTER', 'PRIMARY') == 'CLEAR'
    )  # Determine if wfss or imaging
    bm = 'median' if imaging else 'wfssbkg'  # Background method
    fbc = True
    aff = True if imaging else False  # Apply flat field

    # Define Detector 1 steps
    steps = dict(
        persistence=dict(skip=True),  # Not implemented
        jump=dict(pre_hooks=[cjs], rejection_threshold=5.0),
        clean_flicker_noise=dict(
            skip=False,
            fit_by_channel=fbc,
            background_method=bm,
            apply_flat_field=aff,
            n_sigma=3.0,
        ),
    )

    # Run the pipeline
    stage1 = Detector1Pipeline.call(file, steps=steps)

    # Save to file
    stage1.save(out)

    return

    # Subtract background
    background_filename = cjs.get_reference_file(stage1, 'wfssbkg')
    background_image = cfn._read_image_file(stage1, background_filename, 'image')
    mask, _ = cfn._make_scene_mask(
        None, stage1, False, background_image, 3.0, False, True, False
    )
    background = cfn.background_level(
        stage1.data,
        mask,
        background_method='wfssbkg',
        background_image=background_image,
    )
    stage1.data -= background

    # Flat Field
    steps = dict(
        bkg_subtract=dict(skip=True),
        assign_wcs=dict(skip=True),
        flat_field=dict(skip=False),
        photom=dict(skip=True),
        resample=dict(skip=True),
    )
    stage2 = Image2Pipeline.call(stage1, steps=steps)[0]
    stage2.save(file.replace('_uncal.fits', '_cal.fits'))


if __name__ == '__main__':
    main()
