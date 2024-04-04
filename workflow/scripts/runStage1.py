#! /usr/bin/env python

# Import packages
import os
import sys
import glob
import json
import shutil
import argparse

# Multiprocessing
from multiprocessing import cpu_count,Pool
from threadpoolctl import threadpool_limits

# JWST Pipeline
import jwst
from columnjump import  ColumnJumpStep
from jwst.pipeline import Detector1Pipeline
from snowblind import SnowblindStep, JumpPlusStep
from jwst.step import JumpStep, RampFitStep, AssignWcsStep


# Detector 1 Pipeline
@threadpool_limits.wrap(limits=1, user_api='blas')
def cal(file,out):

    print(f'Processing {file}')

    # Define Detector 1 steps (skip everything before jump)
    steps = dict(
        persistence=dict(
            save_trapsfilled=False # Don't save trapsfilled file
        ),
        jump=dict(
            skip=True,
        ),
        ramp_fit=dict(
            skip=True,
        ),
        gain_scale=dict(
            skip=True,
        )
    )

    # Run the pipeline up until the jump step
    dark = Detector1Pipeline.call(file, steps=steps)

    # Custom Column Jump
    cjump = ColumnJumpStep.call(
        dark,
        nsigma1jump=5.00,
        nsigma2jumps=5
    )

    # Jump step
    jump = JumpStep.call(
        cjump,
        flag_4_neighbors=True,
        expand_large_events=False, # False if using snowblind
        min_jump_to_flag_neighbors=20,
        rejection_threshold=5.0,
        after_jump_flag_time1=0
    )

    # Flag Snowballs w/ Snowblind
    sblind = SnowblindStep.call(
        jump,
        min_radius=3,
        after_jumps=5
    )

    # Jump Plus
    jplus = JumpPlusStep.call(
        sblind
    )

    # Ramp Fit
    rate,_ = RampFitStep.call(
        jplus
    )

    # Save results
    rate.save(out)
    
    print(f'Finished {file}')

    return

# Run pipeline in parallel
if __name__ == '__main__':

    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=(cpu_count() - 2))
    parser.add_argument('--scratch', action='store_true')
    args = parser.parse_args()

    # Get paths
    main = os.getcwd()
    uncal = os.path.join(main,'UNCAL')
    rate = os.path.join(main,'RATE')

    # Restart from scratch
    if not os.path.exists(rate):
        os.mkdir(rate)
    if args.scratch:
        shutil.rmtree(rate)
        os.mkdir(rate)

    # List of files to process
    files = [(f,f.replace(uncal,rate).replace('uncal','rate')) for f in glob.glob(os.path.join(uncal,'*uncal.fits')) if not os.path.exists(os.path.join(rate,os.path.basename(f).replace('uncal','rate')))]

    # Print version
    print(f'jwst:{jwst.__version__}')

    # Multiprocessing
    pool = Pool(processes=args.cpu)
    pool.starmap_async(cal,files,chunksize=1)
    pool.close()
    pool.join()
