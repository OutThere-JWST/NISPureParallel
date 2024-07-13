#! /usr/bin/env python

# Import packages
import os
import argparse
from astropy.table import Table

# Multiprocessing
from multiprocessing import cpu_count, Pool
from threadpoolctl import threadpool_limits

# JWST Pipeline
import jwst
from columnjump import ColumnJumpStep
from jwst.pipeline import Detector1Pipeline
from snowblind import SnowblindStep, JumpPlusStep
from jwst.step import JumpStep, RampFitStep

# Run pipeline in parallel
def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('fieldname', type=str)
    parser.add_argument('--ncpu', type=int, default=(cpu_count() - 2))
    parser.add_argument('--scratch', action='store_true')
    args = parser.parse_args()
    fname = args.fieldname

    # Print version and step
    print(f'Running Stage 1 {fname}')
    print(f'jwst:{jwst.__version__}')

    # Get paths
    main = os.getcwd()
    uncal = os.path.join(main, 'UNCAL')
    rate = os.path.join(main, 'RATE')

    # Get list of products
    prods = Table.read(os.path.join(main, 'FIELDS', 'field-prods.fits'), fname)
    files = prods['productFilename']

    # If not starting from scratch, remove files that are done
    if not args.scratch:
        files = [
            f
            for f in files
            if not os.path.exists(os.path.join(rate, f.replace('uncal', 'rate')))
        ]

    # Get list of inputs
    inputs = [
        (os.path.join(uncal, f), os.path.join(rate, f.replace('uncal', 'rate')))
        for f in files
    ]

    # Multiprocessing
    with Pool(processes=args.ncpu) as pool:
        pool.starmap(cal, inputs, chunksize=1)
        pool.close()
        pool.join()

# Detector 1 Pipeline
@threadpool_limits.wrap(limits=1, user_api='blas')
def cal(file, out):
    print(f'Processing {file}')

    # Define Detector 1 steps (skip everything before jump)
    steps = dict(
        persistence=dict(
            save_trapsfilled=False  # Don't save trapsfilled file
        ),
        jump=dict(
            skip=True,
        ),
        ramp_fit=dict(
            skip=True,
        ),
        gain_scale=dict(
            skip=True,
        ),
    )

    # Run the pipeline up until the jump step
    dark = Detector1Pipeline.call(file, steps=steps)

    # Custom Column Jump
    cjump = ColumnJumpStep.call(dark, nsigma1jump=5.00, nsigma2jumps=5)

    # Jump step
    jump = JumpStep.call(
        cjump,
        flag_4_neighbors=True,
        expand_large_events=False,  # False if using snowblind
        min_jump_to_flag_neighbors=20,
        rejection_threshold=5.0,
        after_jump_flag_time1=0,
    )

    # Flag Snowballs w/ Snowblind
    sblind = SnowblindStep.call(jump, min_radius=3, after_jumps=5)

    # Jump Plus
    jplus = JumpPlusStep.call(sblind)

    # Ramp Fit
    rate, _ = RampFitStep.call(jplus)

    # Save results
    rate.save(out)

    print(f'Finished {file}')

    return

if __name__ == '__main__':
    main()