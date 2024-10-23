#! /usr/bin/env python

# Import packages
import os
import argparse

# Multiprocessing
from threadpoolctl import threadpool_limits

# JWST Pipeline
import jwst
from columnjump import ColumnJumpStep
from jwst.pipeline import Detector1Pipeline

# Set ColumnJumpStep Paramters
columnjump = ColumnJumpStep()
columnjump.nsigma1jump = 5.0
columnjump.nsigma2jumps = 5.0


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

    # Define Detector 1 steps (skip everything before jump)
    steps = dict(
        persistence=dict(skip=True),  # Not implemented
        dark_current=dict(post_hooks=[columnjump]),  # ColumnJump after Dark
        jump=dict(rejection_threshold=5.0),  # 5 sigma rejection in jump
        clean_flicker_noise=dict(skip=True),  # Skip for now
    )

    # Run the pipeline up until the jump step
    rate = Detector1Pipeline.call(file, steps=steps)

    # Save results
    rate.save(out)

    print(f'Finished {out}')

    return


if __name__ == '__main__':
    main()
