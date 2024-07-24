#! /usr/bin/env python

# Import packages
import os
import argparse
import subprocess
from threadpoolctl import threadpool_limits

# JWST Pipeline
import jwst
from jwst.step import RampFitStep
from columnjump import ColumnJumpStep


def main():
    # Parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('uncal', type=str)
    parser.add_argument('--scratch', action='store_true')
    args = parser.parse_args()
    uncal = args.uncal

    # Get output path
    rate = os.path.join('RATE', os.path.basename(uncal).replace('uncal', 'rate'))

    # If exists and not forcing from scratch, skip
    if os.path.exists(rate) and not args.scratch:
        return

    # Run pipeline
    cal(uncal, rate)


# Detector 1 Pipeline
@threadpool_limits.wrap(limits=1, user_api='blas')
def cal(file, out):
    # Print version
    print(f'Processing {file} with jwst: {jwst.__version__}')

    # Steps to process
    steps = [
        'group_scale',
        'dq_init',
        'saturation',
        'superbias',
        'refpix',
        'linearity',
        'dark_current',
        'charge_migration',
        'column',
        'jump',
    ]

    # Arguements
    args = {
        'jump': {
            'rejection_threshold': 5.0,
        },
    }

    # Keep track of original file
    orig = file

    # Loop over steps
    for step in steps:
        # Special handling of column jump
        if step == 'column':
            cjump = ColumnJumpStep.call(file, nsigma1jump=5.00, nsigma2jumps=5)
            cjump.save(orig.replace('uncal', step))

        # Run step
        else:
            # Process arguments
            process = ['strun', step, file, '--output_dir', 'UNCAL', '--suffix', step]

            # Ignore sufficx for jump
            if step == 'jump':
                process = process[:-2]

            # Additional arguments
            if step in args:
                for k, v in args[step].items():
                    process.extend(['--' + k, str(v)])

            # Run step
            subprocess.run(process)

        # Remove old file unless it is the original
        if file != orig:
            print('Removing', file)
            os.remove(file)

        # Update file
        file = orig.replace('uncal', step)

    # Ramp Fit
    file = file.replace('jump', 'column_jump')
    rate, _ = RampFitStep.call(file)

    # Remove old file
    os.remove(file)

    # Save results
    rate.save(out)

    print(f'Finished {out}')

    return


if __name__ == '__main__':
    main()
