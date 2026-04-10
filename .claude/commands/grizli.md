# Grizli Navigation Guide

This is a targeted navigation guide for the grizli fork vendored in this repository.
Use it to orient yourself before reading source files — it tells you *where to look*, not everything you'll find.

## Fork Identity

- **Fork**: `TheSkyentist/grizli` (based on `gbrammer/grizli`)
- **Pinned revision**: check `pixi.toml` under `[feature.grizli.pypi-dependencies]` → `grizli = { git = ..., rev = '...' }` for the current pin
- **Source location**: `vendor/grizli/` (gitignored — local only; clone with the rev from `pixi.toml`:)

```bash
git clone https://github.com/TheSkyentist/grizli.git vendor/grizli
cd vendor/grizli && git checkout <rev from pixi.toml>
```

## Quick Reference

| Pipeline stage | Script | grizli modules used |
|---|---|---|
| Preprocess | `workflow/scripts/preprocess.py` | `jwst_utils`, `utils`, `aws.visit_processor`, `pipeline.auto_script`, `prep` |
| Mosaic | `workflow/scripts/mosaic.py` | `utils`, `aws.visit_processor`, `pipeline.auto_script` |
| Contamination | `workflow/scripts/contamination.py` | `utils`, `aws.visit_processor`, `pipeline.auto_script` |
| Extract | `workflow/scripts/extract.py` | `multifit` |
| Redshift fit | `workflow/scripts/redshiftFit.py` | `fitting`, `pipeline.auto_script` |
| Setup | `resources/scripts/setupGrizli.py` | `utils` |

---

## Module Details

### `grizli/pipeline/auto_script.py`
**Path**: `vendor/grizli/grizli/pipeline/auto_script.py` (~6,770 lines)

End-to-end pipeline orchestration. The main entry point for high-level reductions.

| Symbol | Called in | Purpose |
|---|---|---|
| `auto_script.parse_visits(field_root, RAW_PATH)` | `preprocess.py:90` | Parses FITS headers into visit/group structure |
| `auto_script.grism_prep(field_root, PREP_PATH, EXTRACT_PATH, ...)` | `contamination.py:53` | Builds contamination model (GrismFLT objects) from direct images |
| `auto_script.field_rgb(root, force_rgb, rgb_scl, ...)` | `mosaic.py:116` | Generates color PNG from drizzled mosaics |
| `auto_script.make_filter_combinations(root, filter_combinations, ...)` | `mosaic.py:160` | Combines filter images into a detection image |
| `auto_script.multiband_catalog(field_root, detection_filter, ...)` | `mosaic.py:166` | Runs SExtractor/SEP and produces `{root}-ir.cat.fits` |
| `auto_script.generate_fit_params(pline, field_root, ...)` | `redshiftFit.py:53` | Writes `fit_args.npy` used by `fitting.run_all_parallel` |

---

### `grizli/aws/visit_processor.py`
**Path**: `vendor/grizli/grizli/aws/visit_processor.py` (~3,572 lines)

Despite the `aws` subpackage name, this module handles visit-level processing and is used here entirely *without* AWS infrastructure (the `with_db=False`, `s3output=None`, `sync=False` flags disable cloud paths).

| Symbol | Called in | Purpose |
|---|---|---|
| `visit_processor.ROOT_PATH` (attribute) | `preprocess.py:143` | Set to `FIELDS/` so processor finds files locally |
| `visit_processor.process_visit(fname, tab, prep_args, ...)` | `preprocess.py:144` | Runs Stage 2 reduction + drizzling for one field's visits |
| `visit_processor.res_query_from_local(files)` | `mosaic.py:52`, `contamination.py:153` | Builds association table from local `*rate.fits` headers |
| `visit_processor.cutout_mosaic(fname, res, ir_wcs, ...)` | `mosaic.py:62` | Drizzles direct-image mosaics for a field |

---

### `grizli/utils.py`
**Path**: `vendor/grizli/grizli/utils.py` (~13,865 lines)

Large general-purpose utility module. Only a small subset is used here.

| Symbol | Called in | Purpose |
|---|---|---|
| `utils.make_maximal_wcs(files, pixel_scale, pad, ...)` | `mosaic.py:56` | Computes the smallest WCS that encompasses all input files |
| `utils.Unique(array)` | `contamination.py:155` | Convenience class for unique-value indexing (like `np.unique` + boolean masks) |
| `utils.SRegion(footprint)` | `preprocess.py:262` | Parses sky region footprints; provides `.xy` and `.patch()` for plotting |
| `utils.fetch_default_calibs()` | `resources/scripts/setupGrizli.py` | Downloads HST calibration files |
| `utils.fetch_config_files()` | `resources/scripts/setupGrizli.py` | Downloads grizli config files into `GRIZLI_RESOURCES/` |
| `utils.symlink_templates()` | `resources/scripts/setupGrizli.py` | Symlinks SED templates into the grizli resource path |

---

### `grizli/multifit.py`
**Path**: `vendor/grizli/grizli/multifit.py` (~7,727 lines)

Core spectral extraction classes. Used only in `extract.py`.

| Symbol | Called in | Purpose |
|---|---|---|
| `multifit.GroupFLT(grism_files, catalog, cpu_count, ...)` | `extract.py:61` | Loads all GrismFLT files for a field; manages the contamination model |
| `grp.get_beams(id, size, min_mask, min_sens)` | `extract.py:82` | Extracts 2D beam cutouts for a single object across all exposures |
| `multifit.MultiBeam(beams, fcontam, min_sens, ...)` | `extract.py:87` | Combines beams from multiple exposures; fits and cleans spectra |
| `mb.write_master_fits()` | `extract.py:90` | Writes the `*_beam.fits` and `*_stack.fits` products |

---

### `grizli/fitting.py`
**Path**: `vendor/grizli/grizli/fitting.py` (~6,244 lines)

Template-based redshift fitting. Used only in `redshiftFit.py`.

| Symbol | Called in | Purpose |
|---|---|---|
| `fitting.run_all_parallel(id, zr, args_file, verbose)` | `redshiftFit.py:82` | Top-level fitting function; reads `fit_args.npy`, fits redshift via template marginalization, writes `*.row.fits` and `*.full.fits` |

---

### `grizli/jwst_utils.py`
**Path**: `vendor/grizli/grizli/jwst_utils.py` (~5,178 lines)

JWST-specific utilities. Used only in `preprocess.py`.

| Symbol | Called in | Purpose |
|---|---|---|
| `jwst_utils.set_jwst_to_hst_keywords(file, oneoverf_correction, reset)` | `preprocess.py:34` | Rewrites JWST FITS headers to the HST keyword convention that grizli expects; applied to every RATE file before processing |

---

### `grizli/prep.py`
**Path**: `vendor/grizli/grizli/prep.py` (~9,651 lines)

Image preparation and alignment. The `visit_grism_sky` import is present in `preprocess.py` but the call is commented out.

| Symbol | Called in | Purpose |
|---|---|---|
| `prep.visit_grism_sky(grism, column_average, ignoreNA)` | `preprocess.py:21` (imported; call commented out at line 96–99) | Background subtraction for grism exposures — available but not currently active |

---

## What to Ignore

These parts of grizli are **not used** by this pipeline. Do not explore them unless specifically investigating a new feature:

- `grizli/aws/db.py` — PostgreSQL database layer (we run `with_db=False`)
- `grizli/aws/tile_mosaic.py`, `define_mosaics.py`, `field_tiles.py`, `aws_drizzler.py`, `lambda_handler.py` — AWS-specific tiling infrastructure
- `grizli/galfit/` — GALFIT morphology fitting wrapper
- `grizli/model.py` — Low-level grism beam/model objects (used internally by multifit.py; rarely need to touch directly)
- `grizli/catalog.py` — MAST/archive catalog queries
- `grizli/horizons.py` — Solar system object handling
- `grizli/fake_image.py` — Synthetic image generation
- `grizli/tests/` — Test suite
- `grizli/grismconf.py` — Grism throughput/configuration (used internally)
- `grizli/pipeline/photoz.py`, `summary.py` — Not called by any script here
