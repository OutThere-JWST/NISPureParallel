#! /usr/bin/env python

"""
Standalone trace measurement script.

Consolidates all required code from src/ (constants, config_functions,
measure_traces) into a single self-contained file that runs
run_trace_measurement on a field, iterating over all combinations of
filter (grism), pupil (blocking filter), and spectral order.

Usage
-----
    python trace.py <fieldname>
"""

import argparse
import glob
import os
import warnings

import matplotlib

matplotlib.use('Agg')

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from jwst import datamodels
from jwst.step import AssignWcsStep

warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────
# CONSTANTS  (from src/__init__.py)
# ───────────────────────────────────────────────────

PUPILS = ['F090W', 'F115W', 'F150W', 'F200W']

GRISMS = ['GR150C', 'GR150R']

ORDERS = ['+1', '-1', '+2']

ORDER_MAG_LIMITS = {'+1': 24, '-1': 23, '+2': 22.5}

WAVELENGTH_GRIDS = {
    'F090W': np.arange(0.796, 1.005 + 0.05, 0.05),
    'F115W': np.arange(1.013, 1.283 + 0.05, 0.05),
    'F150W': np.arange(1.330, 1.671 + 0.05, 0.05),
    'F200W': np.arange(1.751, 2.226 + 0.05, 0.05),
}

MIN_FRACTION_ON_DET = 0.2

N_TRACE_SAMPLES = 11


# ───────────────────────────────────────────────────
# HELPER FUNCTIONS  (from src/config_functions.py)
# ───────────────────────────────────────────────────


def find_step_derivative(cutout, bg_npix=10, smooth_window=5):
    """Detect a rising edge using a smoothed derivative."""
    profile = np.sum(cutout[:, :50], axis=0)

    prof = np.asarray(profile, dtype=float)
    n = prof.size
    x = np.arange(n)

    # NaN handling
    isnan = np.isnan(prof)
    if isnan.any():
        try:
            prof[isnan] = np.interp(x[isnan], x[~isnan], prof[~isnan])
        except Exception:
            prof[isnan] = 0.0

    # Background subtract
    bg_npix = min(bg_npix, n // 4)
    bg = np.median(prof[:bg_npix])
    prof_norm = prof - bg

    # Simple smoothing with a symmetric kernel (pure NumPy)
    if smooth_window > 1:
        k = np.ones(smooth_window) / smooth_window
        prof_smooth = np.convolve(prof_norm, k, mode='same')
    else:
        prof_smooth = prof_norm

    # Central-difference derivative
    deriv = np.zeros_like(prof_smooth)
    deriv[1:-1] = 0.5 * (prof_smooth[2:] - prof_smooth[:-2])

    # Rising edge => largest positive derivative
    return np.argmax(deriv)


def fit_trace_undersampled(spectrum, iter=3, init_sigma=0.6):
    """Fit a linear trace to a 2D spectrum using a pixel-integrated Gaussian.

    The cross-dispersion profile at each column is modelled as a Gaussian
    (integrated over each pixel) whose centre follows a straight line
    ``y_centre(x) = m*x + b``.  Because the PSF is undersampled, the
    pixel-integrated form is essential.  The FWHM is constrained to be
    >= 1.3 pixels.

    Per-column amplitudes are profiled out analytically, so the optimiser
    searches only over ``(m, b, sigma)`` — three parameters total.

    After fitting, iterative one-sided background subtraction is performed.

    Parameters
    ----------
    spectrum : array-like, shape (ny, nx)
        2-D spectral image.
    iter : int, optional
        Number of background-subtraction iterations (default 3).
    init_sigma : float, optional
        Initial guess for the Gaussian sigma in pixels.  Default 0.6.

    Returns
    -------
    bg_end : ndarray, shape (ny, nx)
        Accumulated background estimate.
    trace : `~specreduce.tracing.ArrayTrace`
        Fitted trace object compatible with specreduce.
    """
    from astropy.modeling.polynomial import Polynomial1D
    from scipy.optimize import minimize
    from scipy.special import erf
    from specreduce.background import Background
    from specreduce.tracing import ArrayTrace

    data = np.asarray(spectrum, dtype=float)
    ny, nx = data.shape
    y_pix = np.arange(ny, dtype=float)
    x_pix = np.arange(nx, dtype=float)

    # sigma_min from FWHM >= 1.3 px
    min_sigma = 1.3 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def _integrated_gauss(y, mu, sigma):
        """Integral of Gaussian from y-0.5 to y+0.5."""
        s = np.sqrt(2.0) * sigma
        return 0.5 * (erf((y + 0.5 - mu) / s) - erf((y - 0.5 - mu) / s))

    def _objective(params):
        m, b, raw_sigma = params
        sigma = max(raw_sigma, min_sigma)

        centres = m * x_pix + b  # (nx,)
        G = _integrated_gauss(y_pix[:, None], centres[None, :], sigma)  # (ny, nx)

        # Optimal per-column amplitude: A = (G . data) / (G . G)
        GG = np.sum(G * G, axis=0)  # (nx,)
        Gd = np.nansum(G * data, axis=0)  # (nx,)
        safe = GG > 1e-30
        A = np.where(safe, Gd / np.where(safe, GG, 1.0), 0.0)

        model = A[None, :] * G
        return np.nansum((data - model) ** 2)

    # ---- initial guess: collapse along dispersion to find peak row ----
    col_profile = np.nansum(data, axis=1)
    b0 = float(np.argmax(col_profile))
    m0 = 0.0
    sigma0 = max(init_sigma, min_sigma)

    res = minimize(
        _objective,
        [m0, b0, sigma0],
        method='Nelder-Mead',
        options=dict(maxiter=20_000, xatol=1e-5, fatol=1e-6),
    )

    m_fit, b_fit, sigma_raw = res.x
    sigma_fit = max(sigma_raw, min_sigma)  # noqa: F841

    trace_y = m_fit * x_pix + b_fit
    trace = ArrayTrace(spectrum, trace_y)
    trace.trace_model_fit = Polynomial1D(degree=1, c0=b_fit, c1=m_fit)

    # ---- iterative background subtraction ----
    bkg_sep = -4
    bkg_width = 1
    tmp = np.copy(data)
    bg_end = np.zeros_like(data)
    for _ in range(iter):
        bg = Background.one_sided(
            tmp, trace, bkg_sep, statistic='median', width=bkg_width
        )
        tmp = bg.sub_image(tmp).data
        bg_end += bg.bkg_image(spectrum).flux.data

    return bg_end, trace


def in_frame(wcs, ra, dec, pad_x=100, pad_y=100, image_size_x=2048, image_size_y=2048):
    """Return boolean mask of sources inside the detector footprint."""
    x, y, _, _ = wcs(ra, dec, ra, dec)
    ii_in_footprint = (
        (x > pad_x)
        & (x < (image_size_x - pad_x))
        & (y > pad_y)
        & (y < (image_size_y - pad_y))
    )
    if np.sum(ii_in_footprint) == 0:
        print(
            'Warning: no sources fall within the footprint '
            'of the grism image after padding.'
        )
    return ii_in_footprint


# ───────────────────────────────────────────────────
# TRACE MEASUREMENT  (from src/measure_traces.py)
# ───────────────────────────────────────────────────


def show_cutout_with_trace(
    cutout,
    x_trace,
    y_trace,
    x0,
    y0,
    filename,
    obj_index,
    x_det=None,
    y_det=None,
    vmin=None,
    vmax=None,
    cmap='gray',
    edge1=None,
    edge2=None,
    model=None,
    mag=None,
    disp_axis='x',
):
    """Plot cutout with CRDS trace (red) and measured polynomial (cyan)."""

    if vmin is None:
        vmin = np.nanpercentile(cutout, 1)
    if vmax is None:
        vmax = np.nanpercentile(cutout, 99)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(cutout, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # CRDS trace (in cutout coords)
    xt = x_trace - x0
    yt = y_trace - y0
    ax.plot(xt, yt, 'r-', lw=1, label='CRDS trace')
    ax.scatter(xt, yt, c='r', s=10)

    # Fitted polynomial
    if model is not None:
        if disp_axis == 'x':
            x_idx = np.arange(edge2, edge1, 2)
            y_fit = model(x_idx)
            ax.plot(x_idx, y_fit, '-c', lw=1.5, label='Fitted trace')

            if edge1 is not None:
                ax.axvline(edge1, color='cyan', ls='--')
            if edge2 is not None:
                ax.axvline(edge2, color='magenta', ls='--')
        else:
            y_idx = np.arange(edge2, edge1, 2)
            x_fit = model(y_idx)
            ax.plot(x_fit, y_idx, '-c', lw=1.5, label='Fitted trace')

            if edge1 is not None:
                ax.axhline(edge1, color='cyan', ls='--')
            if edge2 is not None:
                ax.axhline(edge2, color='magenta', ls='--')

    title = f'{os.path.basename(filename)} obj={obj_index}'
    if mag is not None and np.isfinite(mag):
        title += f' mag={mag:.1f}'
    if x_det is not None:
        title += f' det=({x_det:.1f},{y_det:.1f})'

    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig, ax


def select_sources(
    cat, order, mag_limit=None, size_range=None, ellip_limit=None, neighbor_radius=None
):
    """Select sources from a catalog based on magnitude, size, ellipticity,
    and isolation."""
    # Magnitude selection
    mag_est = cat['MAG_AUTO']

    if mag_limit is None:
        mag_limit = ORDER_MAG_LIMITS[order]
    ii_bright = mag_est < mag_limit

    # Size selection
    if size_range is not None:
        min_size, max_size = size_range
        ii_size = (cat['FLUX_RADIUS'] > min_size) & (cat['FLUX_RADIUS'] < max_size)
    else:
        ii_size = np.ones(len(cat), dtype=bool)

    # Ellipticity selection
    if ellip_limit is not None:
        ell = 1 - (cat['B_IMAGE'] / cat['A_IMAGE'])
        ii_ellip = ell < ellip_limit
    else:
        ii_ellip = np.ones(len(cat), dtype=bool)

    # Isolation selection
    if neighbor_radius is not None:
        ra_val = np.array(cat['RA'], dtype=float)
        dec_val = np.array(cat['DEC'], dtype=float)
        coords = SkyCoord(ra_val * u.deg, dec_val * u.deg)
        idx, sep2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
        ii_iso = sep2d.arcsec > neighbor_radius
    else:
        ii_iso = np.ones(len(cat), dtype=bool)

    ii_final = ii_bright & ii_size & ii_ellip & ii_iso
    selected = cat[ii_final]
    print(f'[select_sources] order {order}: final N={len(selected)}')

    # Returns the mask rather than the trimmed catalog
    return ii_final


def make_plot_dir(input_dir, root, filter, pupil, order):
    """Return (and create) directory for saving trace sanity-check plots."""
    if input_dir is None:
        raise ValueError('input_dir must be a valid directory string, not None.')
    d = os.path.join(
        input_dir, 'TRACE_PLOTS', f'{str(root)}_{str(filter)}_{str(pupil)}_{order}'
    )

    print(f'Saving trace plots to: {d}')

    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------
# MEASURE TRACES FOR ONE EXPOSURE
# ---------------------------------------------------
def measure_traces_one_exposure(
    filename, cat, outname=None, order='+1', mag_limit=None
):
    """Measure traces for all orders for one grism exposure."""

    print(f'\n=== Processing exposure: {filename} ===')

    # JWST WCS / transforms
    dm = datamodels.open(filename)
    dm = AssignWcsStep.call(dm)
    tform = dm.meta.wcs.get_transform('detector', 'grism_detector')[-1]
    wcs = dm.meta.wcs.get_transform('world', 'detector')

    hdr = fits.getheader(filename, 0)
    pupil = hdr.get('PUPIL', 'UNKNOWN')
    filter_name = hdr.get('FILTER', 'UNKNOWN')  # GR150C or GR150R

    if outname is None:
        out_base = os.path.splitext(filename)[0]
        outname = f'{out_base}_{filter_name}_{pupil}_traces.txt'

    dir = os.path.dirname(filename)
    root = os.path.splitext(filename)[0].split('/')[-1]

    # Pick wavelength grid
    if pupil not in WAVELENGTH_GRIDS:
        print(f'[WARNING] Skipping file {filename}: unsupported pupil {pupil}')
        return

    wavelength_grid = WAVELENGTH_GRIDS[pupil]

    if mag_limit is None:
        mag_limit = ORDER_MAG_LIMITS[order]

    # Create output table (overwrite)
    with open(outname, 'w') as fh:
        fh.write(f'# input={os.path.basename(filename)} pupil={pupil}\n')
        fh.write('# id ra dec order x_det y_det x_det_trace y_det_trace\n')

    grism_image = fits.getdata(filename, 1)
    ny, nx = grism_image.shape  # (y, x)

    print(f'→ Order {order}')

    ii_bright = select_sources(cat, order, mag_limit=mag_limit)
    ii_in_footprint = in_frame(
        wcs,
        cat['RA'],
        cat['DEC'],
        pad_x=100,
        pad_y=100,
        image_size_x=nx,
        image_size_y=ny,
    )

    ii_keep = ii_in_footprint & ii_bright
    cat_sel = cat[ii_keep]
    print(f'  Selected bright: {len(cat_sel)} sources')

    plot_dir = make_plot_dir(
        dir, root=root, filter=filter_name, pupil=pupil, order=order
    )

    if 'C' in filter_name:
        disp_axis = 'x'
    elif 'R' in filter_name:
        disp_axis = 'y'

    # Define intended bounding box around the full trace
    if order == '+1':
        pad_x, pad_y = 20, 10
    else:
        pad_x, pad_y = 20, 15

    for row in cat_sel:
        ra, dec = row['RA'], row['DEC']
        mag = row['MAG_AUTO']
        obj_id = row['NUMBER']

        # Detector position of the object
        x_det, y_det, _, _ = wcs(ra, dec, ra, dec)

        # CRDS trace for wavelength grid
        x_tr, y_tr, _, _, _ = tform(x_det, y_det, wavelength_grid, order)

        # Detect dispersion axis
        dx_span = np.nanmax(x_tr) - np.nanmin(x_tr)  # noqa: F841
        dy_span = np.nanmax(y_tr) - np.nanmin(y_tr)  # noqa: F841

        x0_raw = int(np.nanmin(x_tr) - pad_x)
        x1_raw = int(np.nanmax(x_tr) + pad_x)
        y0_raw = int(np.nanmin(y_tr) - pad_y)
        y1_raw = int(np.nanmax(y_tr) + pad_y)

        # Clip box to detector, allowing partial traces
        x0 = max(0, x0_raw)
        y0 = max(0, y0_raw)
        x1 = min(nx, x1_raw)
        y1 = min(ny, y1_raw)

        # Fraction of the intended box that remains on detector
        width_raw = max(1, x1_raw - x0_raw)
        height_raw = max(1, y1_raw - y0_raw)
        width_clipped = max(0, x1 - x0)
        height_clipped = max(0, y1 - y0)

        frac_x = width_clipped / width_raw
        frac_y = height_clipped / height_raw

        if (width_clipped < 10) or (height_clipped < 10):
            print(
                f'  [WARN] Object {obj_id}: clipped subimage too small '
                f'({width_clipped}x{height_clipped}) → skipping'
            )
            continue

        if (frac_x < MIN_FRACTION_ON_DET) or (frac_y < MIN_FRACTION_ON_DET):
            print(
                f'  [WARN] Object {obj_id}: <{MIN_FRACTION_ON_DET:.2f} '
                'of trace box on detector → skipping'
            )
            continue

        # Extract cutout from clipped box
        cutout = grism_image[y0:y1, x0:x1]

        # Fit trace using FitTrace on cutout (or its transpose)
        work = cutout if disp_axis == 'x' else cutout.T

        # -------------------------------
        # EDGE DETECTION (BOTH AXES)
        # -------------------------------
        pad_disp = pad_x if disp_axis == 'x' else pad_y

        # Find edge on the "right" side by flipping horizontally
        edge_start_flipped = find_step_derivative(np.fliplr(work)[:, :50], bg_npix=10)
        edge_start = work.shape[1] - 1 - edge_start_flipped

        # Find edge on the "left" side
        edge_end = find_step_derivative(work[:, :50], bg_npix=10)

        # Skip objects with edges too far from expected positions
        if ~(
            (edge_start > work.shape[1] - pad_disp * 1.5)
            and (edge_start < work.shape[1] - pad_disp / 2)
            and (edge_end > pad_disp / 2)
            and (edge_end < pad_disp * 1.5)
        ):
            continue

        # Slice between detected edges in the working frame
        sub = work[:, edge_end:edge_start]

        # Basic sanity checks
        if not np.isfinite(sub).any():
            print(f'  [WARN] Object {obj_id}: subimage is fully NaN → skipping')
            continue
        if np.nanmax(sub) == 0 or np.nanstd(sub) == 0:
            print(f'  [WARN] Object {obj_id}: subimage has no signal → skipping')
            continue
        if sub.shape[0] < 10 or sub.shape[1] < 10:
            print(
                f'  [WARN] Object {obj_id}: subimage too small {sub.shape} → skipping'
            )
            continue

        try:
            bg, tr = fit_trace_undersampled(sub, iter=3)
        except Exception as e:
            print(f'  [WARN] FitTrace failed for obj {obj_id}: {e}')
            continue

        # Model returned by fit is defined in the sliced-subimage x-coordinates.
        # We wrap it so callers can use full WORK-frame coordinates.
        model_sub = tr.trace_model_fit
        shift = edge_end

        def model_work(x, m=model_sub, s=shift):
            x = np.asarray(x)
            return m(x - s)

        # -------------------------------
        # SAVE SANITY-CHECK TRACE PLOTS
        # -------------------------------
        # fig, ax = show_cutout_with_trace(
        #     cutout,
        #     x_tr,
        #     y_tr,
        #     x0,
        #     y0,
        #     filename,
        #     obj_id,
        #     x_det=x_det,
        #     y_det=y_det,
        #     edge1=edge_start,
        #     edge2=edge_end,
        #     model=model_work,
        #     mag=mag,
        #     disp_axis=disp_axis,
        # )
        # fig.savefig(f'{plot_dir}/obj_{obj_id:06d}.png', dpi=120)
        # plt.close(fig)

        # -------------------------------
        # WRITE TRACE SAMPLE POINTS
        # -------------------------------
        if disp_axis == 'x':
            # Sample along x (dispersion), model gives y
            idx = np.linspace(edge_end, edge_start - 1, N_TRACE_SAMPLES).astype(int)

            for xx in idx:
                yy = model_work(xx)
                X, Y = xx + x0, yy + y0
                with open(outname, 'a') as fh:
                    fh.write(
                        f'{obj_id} {ra:.6f} {dec:.6f} {order} '
                        f'{x_det:.2f} {y_det:.2f} {X:.2f} {Y:.2f}\n'
                    )

        else:
            # Dispersion is along y in the original cutout.
            idx = np.linspace(edge_end, edge_start - 1, N_TRACE_SAMPLES).astype(int)

            for yy in idx:
                xx = model_work(yy)
                X, Y = xx + x0, yy + y0
                with open(outname, 'a') as fh:
                    fh.write(
                        f'{obj_id} {ra:.6f} {dec:.6f} {order} '
                        f'{x_det:.2f} {y_det:.2f} {X:.2f} {Y:.2f}\n'
                    )


# ---------------------------------------------------
# FIND RATE FILES BY HEADER
# ---------------------------------------------------
def find_rate_files(directory, filter=None, pupil=None):
    """
    Find RATE files for a given field, using FITS headers (PUPIL, FILTER)
    instead of file names.
    """
    if (filter is None) or (pupil is None):
        print("Error: Both 'filter' and 'pupil' must be defined.")
        return []

    path = os.path.join(directory, '*rate.fits')
    files = glob.glob(path)

    selected = []

    for fname in files:
        try:
            hd = fits.getheader(fname, 0)
        except Exception:
            continue

        file_pupil = hd.get('PUPIL', '')
        file_filter = hd.get('FILTER', '')

        if (filter == file_filter) and (file_pupil == pupil):
            selected.append(fname)

    return selected


# ---------------------------------------------------
# MEASURE TRACES FOR ONE COMBO
# ---------------------------------------------------
def run_trace_measurement(
    directory, filter=None, pupil=None, order='+1', cat=None, mag_limit=None
):
    """
    Run trace measurement for all RATE exposures in ``directory`` that match
    exact FITS header FILTER and PUPIL values.

    Parameters
    ----------
    directory : str
        Path to the directory containing ``*rate.fits`` files.
    filter : str
        Grism name to match against the FITS FILTER header
        (e.g. ``'GR150C'``).
    pupil : str
        Blocking-filter name to match against the FITS PUPIL header
        (e.g. ``'F200W'``).
    order : str
        Spectral order (``'+1'``, ``'-1'``, or ``'+2'``).
    cat : `~astropy.table.Table`
        Source catalog (already loaded).
    mag_limit : float or None
        Override for the default per-order magnitude limit.
    """
    if not directory or not isinstance(directory, str):
        raise ValueError('`directory` must be a non-empty string.')
    if not filter or not isinstance(filter, str):
        raise ValueError("`filter` must be a non-empty string (e.g. 'GR150R').")
    if not pupil or not isinstance(pupil, str):
        raise ValueError('`pupil` must be a non-empty string.')
    if cat is None:
        raise ValueError('`cat` must be an astropy Table.')

    rate_files = find_rate_files(directory=directory, filter=filter, pupil=pupil)

    if len(rate_files) == 0:
        return

    print(
        f'\nFound {len(rate_files)} exposures for FILTER={filter} / '
        f'PUPIL={pupil} / order={order}'
    )

    for fname in rate_files:
        measure_traces_one_exposure(
            filename=fname, cat=cat, outname=None, order=order, mag_limit=mag_limit
        )


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run trace measurement on a NIRISS pure-parallel field.'
    )
    parser.add_argument('fieldname', type=str, help='Name of the field.')
    args = parser.parse_args()
    fname = args.fieldname

    print(f'Tracing {fname}')

    # Derive paths (same layout as other workflow scripts)
    main_dir = os.getcwd()
    fields = os.path.join(main_dir, 'FIELDS')
    home = os.path.join(fields, fname)
    rate_dir = os.path.join(home, 'RATE')
    prep = os.path.join(home, 'Prep')
    cat_path = os.path.join(prep, f'{fname}-ir.cat.fits')

    # Load catalog once
    cat = Table.read(cat_path)
    print(f'Loaded catalog: {len(cat)} sources from {cat_path}')

    # Iterate over every grism x blocking-filter x order combination
    for grism in GRISMS:
        for pupil in PUPILS:
            for order in ORDERS:
                print(f'\n=== {grism} / {pupil} / order {order} ===')
                run_trace_measurement(
                    directory=rate_dir, filter=grism, pupil=pupil, order=order, cat=cat
                )

    print('\n=== Tracing complete ===')


if __name__ == '__main__':
    main()
