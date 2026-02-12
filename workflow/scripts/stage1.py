#! /usr/bin/env python

import argparse
import os

import jwst
import numpy as np
from astropy.io import fits
from astropy.modeling import Fittable1DModel, Parameter, fitting, models
from astropy.stats import SigmaClip
from columnjump import ColumnJumpStep
from jwst.pipeline import Detector1Pipeline
from jwst.ramp_fitting.ramp_fit_step import RampFitStep
from photutils.background import Background2D, MedianBackground
from threadpoolctl import threadpool_limits


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
    imaging = fits.getval(file, 'FILTER', 'PRIMARY') == 'CLEAR'
    fit_by_channel = False
    fit_bkg = False

    # Define Detector 1 steps
    steps = dict(
        persistence=dict(skip=True),  # Not implemented
        jump=dict(pre_hooks=[cjs], rejection_threshold=5.0),
        ramp_fit=dict(skip=True),
        gain_scale=dict(skip=True),
    )

    # Run the pipeline up to jump step
    jump = Detector1Pipeline.call(file, steps=steps)

    # Run initial ramp fitting
    rfs = RampFitStep()
    rate_init, _ = rfs.run(jump)

    # Get the number of groups and ints and array dimension
    Ngroups = jump.meta.exposure.ngroups
    Nints = jump.meta.exposure.nints
    N = jump.data.shape[-1]

    # Get background
    bkg_im = fits.getdata(
        cjs.get_reference_file(jump, 'flat' if imaging else 'bkg'), 'SCI'
    )

    # Create the generic sigma clipping fitter and model
    model = models.Linear1D(slope=1.0, intercept=0.0, fixed={'intercept': True})
    sigma_clip = SigmaClip(sigma=3, maxiters=5)
    fitter = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(), sigma_clip)

    # Create background mask
    mask = np.full(rate_init.dq.shape, True)
    mask[np.isnan(rate_init.data) | (rate_init.data == 0) | np.isnan(bkg_im)] = False
    fit, outliers = fitter(model, bkg_im[mask], rate_init.data[mask])
    mask[mask] = np.invert(outliers)

    # NaN out mask
    nan_mask = mask.astype(float)
    nan_mask[~mask] = np.nan

    # Calculate difference image
    raw_diffs = np.diff(jump.data, axis=1)
    diffs = raw_diffs * nan_mask

    # Fitting inputs
    x = (bkg_im[mask],)
    X = (np.tile(bkg_im[mask], Nints),)

    if fit_bkg:
        # Get flat image
        flat_im = fits.getdata(cjs.get_reference_file(jump, 'flat'), 'SCI')

        # Compute smooth background
        remainder = rate_init.data
        remainder[mask] -= fit(bkg_im[mask])
        smooth_bkg = Background2D(
            remainder / flat_im,
            box_size=(32, 32),
            filter_size=(5, 5),
            mask=~mask,
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
        ).background
        smooth_bkg *= flat_im

        # Use the two component model
        model = Linear2ImageModel()

        # Add to fitting inputs
        x += (smooth_bkg[mask],)
        X += (np.tile(smooth_bkg[mask], Nints),)

    # Fit and subtract the background
    for j in range(Ngroups - 1):
        y = diffs[:, j, :, :][:, mask].ravel()
        fit, _ = fitter(model, *X, y)
        diffs[:, j, :, :][:, mask] -= fit(*X).reshape((Nints, -1))

    # Unravel the difference data
    Nchannel = 4 if fit_by_channel else 1
    width = diffs.shape[3] // Nchannel
    diffs_unravel = diffs.reshape(Nints, Ngroups - 1, Nchannel, width, N)

    # Compute 1/f noise correction
    oof = np.nan_to_num(np.nanmedian(diffs_unravel, axis=-2), copy=False, nan=0.0)
    correction_unravel = np.zeros_like(diffs_unravel) + oof[..., None, :]
    correction = correction_unravel.reshape(Nints, Ngroups - 1, Nchannel * width, N)

    # Apply correction to the data
    jump.data[:, 1:] = jump.data[:, 0:1] + np.cumsum(raw_diffs - correction, axis=1)

    # Run ramp fitting
    stage1, _ = rfs.run(jump)

    # Save to file
    stage1.save(out)


class Linear2ImageModel(Fittable1DModel):
    """
    Linear model: y = a * x1 + b * x2
    """

    n_inputs = 2
    n_outputs = 1

    a = Parameter(default=1.0)
    b = Parameter(default=1.0)

    linear = True  # Enables use with LinearLSQFitter

    @staticmethod
    def evaluate(x1, x2, a, b):
        """
        Evaluate the model: y = a * x1 + b * x2
        """
        return a * x1 + b * x2

    @staticmethod
    def fit_deriv(x1, x2, a, b):
        """
        Derivatives with respect to parameters a and b.
        """
        return [x1, x2]


if __name__ == '__main__':
    main()
