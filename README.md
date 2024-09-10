## Automated Parallelization of NIRISS WFSS Pure Parallel Data

This repository contains to process all of the available JWST NIRISS Pure Parallel observations from PASSAGE (1571) and OutThere (3383 + 4681). It uses a Pythonic workflow manager called [Snakemake](https://snakemake.readthedocs.io/en/stable/) in order to distribute the parallizeable steps of the workflow and can be scaled to cluster environments. In order to run the workflow yourself, you will need to do the following:
1. [Set Up Grizli](#set-up-grizli)
2. [Set Up NISPureParallel](#set-up-nispureparallel)
3. [Create the SnakeMake environment](#creating-snakemake-environment)
4. [Fetch NIRISS Data](#fetch-niriss-data)
5. [Configure SnakeMake](#configure-snakemake)
6. [SnakeMake!](#snakemake)

### Set Up Grizli
The workflow described here will install all of the necessary software for you, however it does not install configuration/calibration files necessary for much of the analysis. In order to do this I recommend the following:
- Follow the [grizli installation instructions](https://grizli.readthedocs.io/en/latest/grizli/install.html), most importantly the part describing the downloading of configuration files and specifying path variables. Since grizli is pip-installable, I recommend making a venv or conda/mamba environment and pip installing and then downloading the relevant files. Make sure to download the JWST calibrations! 
- You'll also need to set the [relevant CRDS environment variables](https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/reference_files_crds.html). It may also be worth running the pipeline on a single UNCAL file independent of the following steps so that the correct CRDS files are fetched and placed in your specified cache directory.

### Download Additional Calibrations
- This Pipeline requires some additional calibration files to be placed in your grizli configuration directory:
    1. Background Subtraction Files for Imaging and WFSS: https://zenodo.org/records/13741413. Make sure to unpack the non-unflat versions.
    2. (Option) Updated Trace Configurations: https://zenodo.org/records/7628094

### Set Up NISPureParallel
This comes in three steps:
1) Clone this GitHub repository, note that the end products of this process can take up a few TB of disk space.
2) `cd NISPureParallel`
3) Create the snakemake environment: `conda/mamba env create -f snakemake.yaml`. This environment includes the code necessary to run our Snakemake workflow, but also to download the NIRISS data and compute the overlapping regions, etc.

It may also be necessary to clear the snakemake-created environments every once in a while if Grizli or the JWST pipeline are updated. You can do this with `rm -r .snakemake/conda`.

### Set Up NIRISS Fields
This pipeline processes data in `fields` which are computed in the following way:
- If two NIRISS fields overlap (computed using a Cartesian approximation) they are added to the same field.
- If two NIRISS fields are within 30" of each other (computed using a spherical prescription), they are added to the same field. This is to account for the fact that the trace from a galaxy in one NIRISS field may fall into a non-contiguous field some distance away.
- Fields are named based on the constellation they are in along with a numeric index that indicates the relative start of observation of the field (e.g. the first data for field uma-00 arrived before uma-01).
After activating the Snakemake environment, we can run the conveniently provided helper script to compute these fields. Make sure to run this from the top-level directory as such: `python ./resources/scripts/computeFields.py`. This will query all available NIRISS PP data from OutThere and PASSAGE and compute the resultant fields. 

All data are processed in each of their fields. 

### Configure SnakeMake
If you are running this on a cluster (or even on a powerful server) you will likely need to write a Snakemake configuration profile for the workflow. You can see examples of these in the `profiles/` directory.

Essentially this allows Snakemake to submit your SLURM jobs on your behalf. As such it will require basic SLURM information, such as the number of available CPUs per node and the number of jobs Snakemake can submit at once. 

One important note is that due to the relatively high memory usage of the JWST Stage 1 Pipeline, it may be necessary to engage in some manual intervention to limit the number of cores used by the first step in the pipeline.

### Snakemake!
Now we just have to type one command and watch as our worflow is distributed over our cluster:
`snakemake --workflow-profile profiles/your-profile`
And that's it! Snakemake will distribute this workflow over the clusters for you! Two additional flags that you should be aware of:
- `--force-all`: Snakemake can recognize when it needs to re-run a rule in many cases, such as a change in the conda/mamba environment used for the rule, or a change in the Snakemake rule (see below). But it's always good to have a hammer when you want to force a full rerun.
- `--rerun-incomplete`: Sometimes a job/rule will fail, leaving behind bad files. Running with this option will get rid of pesky errors related to this, and force those jobs to rerun. 

If you just want to run a single field, simply provide the relevant field output file as input to the command: `snakemake FIELDS/leo-00/logs/fmap.log`.

If you've edited a step of the pipeline and want to force it to re-run just for that step, you can delete the relevant logfile from the log directory.

Note: For cluster users, you will need to make sure that the required environment modules are loaded in your bashrc/profile. There should be a way to do it with snakemake but I haven't managed to get it to work yet. Importantly, conda and TeX are needed for running this pipeline, which are available to the MPIA Cluster as modules.

## Pipeline
The following is a brief description of Snakemake and the pipeline developed in this repository. All of the relevant data and files are contained within the `workflow` directory. Snakemake begins by reading the Snakefile which describes the rules necessary to create output files based on input files. Snakemake then computes the DAG necessary to create all requested output files and distributes them in parallel where possible. Each rule requires inputs, outputs, and code necessary to produce inputs from outputs. Here I briefly describe the stages of our pipeline:

0) Download Data. The UNCAL images for the field in question is downloaded. If the data exist, it will not re-download them. The download is parallelized.

1) Stage1 Processing. Here we convert UNCAL files for NIRISS into RATE files. Here we use the latest version of the JWST pipeline and implement the ColumnJump step which can offer improvements for NIRISS fields. 

2) Preprocessing. Here the data is pre-processed using `grizli`. In a nutshell, a modified version of the Stage2 pipeline is applied, and dithered frames are drizzled together.

3) Mosaicing. Frames in the same field are mosaiced together and a detection catalog is measured on an IR image made from combining all available direct filters.

4) Contamination Modelling. Based on the direct image and detection catalog, we create a model of the dispersed image that can be used to model the contamination from overlapping traces. 

5) Spectral Extraction. 2D Spectra are extracted from the dispersed images. 

6) Redshift Fitting. 

7) FITSMap. In the final step, the products are collected and a FITSMap is created highlighting all of the results of the data analysis.


## Notes

Some fields have very large input files. Therefore, they will likely fail Stage1 Processing due to memory issues. You can circumvent this by artificially lowering the multiprocessing count for the Stage 1 rule and re-running on just the problematic fields. 