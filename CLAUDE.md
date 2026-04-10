# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an automated Snakemake pipeline for processing JWST NIRISS Wide Field Slitless Spectroscopy (WFSS) Pure Parallel data from the PASSAGE (1571) and OutThere (3383+4681) programs. The pipeline handles the full reduction chain from raw UNCAL files through spectral extraction, redshift fitting, and FITSMap visualization.

## Environment Management

This project uses [pixi](https://pixi.sh) for environment management. There are four isolated pixi environments defined in `pixi.toml`, each used for a specific pipeline stage:

| Environment | Used for |
|---|---|
| `snakemake` | Workflow orchestration |
| `jwst` | Stage 1 JWST pipeline processing |
| `grizli` | Preprocessing, mosaicing, contamination, extraction, redshift fitting |
| `fitsmap` | Final FITSMap creation |

Key environment variables are set in `pixi.toml` under `[activation.env]`:
- `GRIZLI` → `GRIZLI_RESOURCES/`
- `CRDS_PATH` → `CRDS_CACHE/`
- `CRDS_CONTEXT` → pinned to a specific pmap version

## Common Commands

```bash
# Set up grizli configuration files and download configs
pixi run -e grizli grizli-setup

# Compute fields from available data (creates FIELDS/fields.fits)
pixi run compute-fields

# Run the full Snakemake pipeline (local, default profile)
pixi run -e snakemake snakemake --workflow-profile profiles/default

# Run on a cluster (e.g., MPCDF Vera or Swinburne Ngarrgu Tindebeek)
pixi run -e snakemake snakemake --workflow-profile profiles/vera
pixi run -e snakemake snakemake --workflow-profile profiles/ngarrgu-tindebeek

# Run only a single field
pixi run -e snakemake snakemake FIELDS/leo-00/logs/fmap.log

# Force rerun of incomplete jobs
pixi run -e snakemake snakemake --workflow-profile profiles/your-profile --rerun-incomplete

# Force rerun everything
pixi run -e snakemake snakemake --workflow-profile profiles/your-profile --forceall

# Enter the snakemake environment shell
pixi shell -e snakemake
```

To force a specific pipeline stage to re-run for a field, delete the relevant logfile from `FIELDS/{field}/logs/`.

## Pipeline Architecture

### Field Definition
Fields are computed by `resources/scripts/computeFields.py` and stored in `FIELDS/fields.fits` (a multi-extension FITS file). Each HDU extension is a field; fields group NIRISS observations that either spatially overlap or are within 30" of each other. Fields are named by constellation + numeric index (e.g., `uma-00`).

### Snakemake DAG (`workflow/Snakefile`)
The Snakefile reads `FIELDS/fields.fits` to determine field membership and RATE file targets, then defines a linear chain of rules:

```
download → stage1 → preprocess → mosaic → contam → extract → zfit → fmap
```

Each rule's completion is tracked by a logfile under `FIELDS/{field}/logs/`. The CRDS cache is synced at workflow start via an `onstart` hook.

When working with grizli-related code, invoke the `/grizli` skill for module orientation before reading source files.

### Pipeline Scripts (`workflow/scripts/`)
Each script is invoked via the appropriate pixi environment:

| Script | Environment | Purpose |
|---|---|---|
| `fetchCRDS.py` | `jwst` | Sync NIRISS CRDS reference files |
| `download.py` | default | Download UNCAL files |
| `stage1.py` | `jwst` | JWST Detector1Pipeline with ColumnJump + custom 1/f correction |
| `preprocess.py` | `grizli` | Modified Stage 2 + drizzle dithers |
| `mosaic.py` | `grizli` | Mosaic fields, build IR detection catalog |
| `contamination.py` | `grizli` | Build contamination model from direct images |
| `extract.py` | `grizli` | Extract 2D spectra |
| `redshiftFit.py` | `grizli` | Fit photometric/spectroscopic redshifts |
| `makeFitsmap.py` | `fitsmap` | Collect products, build FITSMap |

### Stage 1 Details (`workflow/scripts/stage1.py`)
Stage 1 is custom beyond the standard JWST pipeline:
1. Runs `Detector1Pipeline` through the jump step (skipping `ramp_fit` and `gain_scale`)
2. Applies `ColumnJumpStep` (from `columnjump` package) before jump detection
3. Performs 1/f noise correction by fitting column-median differences against a background reference image
4. Runs `RampFitStep` twice — once to get an initial rate for fitting, then again after applying the 1/f correction

### Cluster Profiles (`profiles/`)
- `default/` — local execution, 2 cores, 8 CPUs per task
- `vera/` — MPCDF Vera (SLURM), 200 jobs max, 72 CPUs/node
- `ngarrgu-tindebeek/` — Swinburne OzSTAR (SLURM), 5000 jobs max, milan/trevor partitions
- `astro-node/` — local server profile

Download and stage1 rules use `group:` directives for batching multiple files into a single SLURM job. Group sizes are configured per-profile via `group-components`.

## Key Directories

- `FIELDS/` — pipeline outputs, one subdirectory per field
- `CRDS_CACHE/` — JWST CRDS reference files (~35GB)
- `GRIZLI_RESOURCES/` — grizli configuration files
- `resources/scripts/` — utility scripts (field computation, HTML generation, S3 sync)
- `test/` — test configuration and reference files for NIRISS WFSS
- `vendor/grizli/` — vendored source of the pinned grizli fork (`TheSkyentist/grizli`); gitignored and local-only — read source files here directly when investigating grizli internals

## Vendored Packages

`vendor/grizli/` contains the full grizli source at the revision pinned in `pixi.toml` (see `[feature.grizli.pypi-dependencies]`). It is gitignored (not committed) and must be cloned locally at the rev specified in `pixi.toml`:

```bash
git clone https://github.com/TheSkyentist/grizli.git vendor/grizli
cd vendor/grizli && git checkout <rev from pixi.toml>
```

Use the `/grizli` skill for a curated map of which modules and symbols are relevant to this pipeline.
