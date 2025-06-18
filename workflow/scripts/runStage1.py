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
from jwst.pipeline import Detector1Pipeline


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
    imaging = fits.getval(file, 'FILTER', 'PRIMARY') != 'CLEAR' # Determine if wfss or imaging
    bm = 'median' if imaging else 'model'  # Background method
    fbc = False #True if imaging else False  # Fit by channel
    aff = True if imaging else False

    # Define Detector 1 steps
    steps = dict(
        persistence=dict(skip=True),  # Not implemented
        jump=dict(pre_hooks=[cjs], rejection_threshold=5.0),
        clean_flicker_noise=dict(
            skip=False, fit_by_channel=fbc, background_method=bm, apply_flat_field=aff
        ),
    )

    # Run the pipeline
    stage1 = Detector1Pipeline.call(file, steps=steps)

    # Save to file
    stage1.save(out)

    return


if __name__ == '__main__':
    main()
