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
from jwst.ramp_fitting import RampFitStep
from jwst.clean_flicker_noise import CleanFlickerNoiseStep


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

    # Define Detector 1 steps up until Ramp Fitting
    steps = dict(
        persistence=dict(skip=True),  # Not implemented
        charge_migration=dict(post_hooks=[cjs]),
        jump=dict(rejection_threshold=5.0),
        clean_flicker_noise=dict(skip=True),
        ramp_fit=dict(skip=True),
    )

    # Run the pipeline up until the jump step
    stage1 = Detector1Pipeline.call(file, steps=steps)

    # Determine if wfss or imaging
    disperser = fits.getval(file, 'FILTER', 'PRIMARY')

    # Decide 1/f correction parameters
    bm = 'median' if disperser == 'CLEAR' else 'model'  # Background method
    fbc = True if disperser == 'CLEAR' else False  # Fit by channel
    skip = False if disperser == 'CLEAR' else True # Skip step for WFSS

    if not skip:

        # Get flat-field
        flat_field = fits.getdata(cjs.get_reference_file(file, 'flat'))
        flat_field[flat_field == 0] = 1  # Ignore zeros

        # Divide out flat-field
        stage1.data /= flat_field

        # Clean Flicker Step
        cfns = CleanFlickerNoiseStep()
        cfns.skip, cfns.fit_by_channel, cfns.background_method = skip, fbc, bm
        stage1 = cfns.run(stage1)

        # Reapply flat field
        stage1.data *= flat_field

    # Ramp Fit Step
    rate, _ = RampFitStep().run(stage1)

    # Save to file
    rate.save(out)

    return


if __name__ == '__main__':
    main()
