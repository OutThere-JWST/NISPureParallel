## Automated Pipeline of NIRISS WFSS Pure Parallel Data

This repository contains to process all of the available JWST NIRISS Pure Parallel observations from PASSAGE (1571) and OutThere (3383 + 4681). It uses an Environment Manager [pixi](https://pixi.sh/latest/) to install software combined with a Pythonic workflow manager called [Snakemake](https://snakemake.readthedocs.io/en/stable/) in order to distribute the parallizeable steps of the workflow and can be scaled to cluster environments. In order to run the workflow yourself, you will need to do the following:

1. [Install Pixi](#install-pixi)
2. [Clone this Repository](#clone-this-repository) 
3. [Configure Grizli and CRDS](#configure-grizli-and-crds)
4. [Set Up Pipeline](#set-up-pipeline)
5. [Pipeline Description](#pipeline-description)
5. [SnakeMake!](#snakemake)
6. [Summary](#summary)

### Install Pixi
This pipeline uses [pixi](https://pixi.sh/latest/) as an environment manager. To learn more about pixi, check out their documentation but most users can install it with:
`curl -fsSL https://pixi.sh/install.sh | bash`

### Clone this Repository
To get started, of course the first step is to clone this repository either over HTTPS or SSH:

`git clone git@github.com:OutThere-JWST/NISPureParallel.git`

`git clone https://github.com/OutThere-JWST/NISPureParallel.git`

This repository includes the following:

- Pixi environment files for installing the necessary software
- Snakemake profiles for different machines/clusters
- Snakemake workflow code for processing NIRISS Pure Parallel Data


### Configure Grizli and CRDS
Those who have installed grizli and/or the JWST pipeline before may be familiar with the process of installing grizli, configuring the relevant environment variables, and downloading the necessary configuration files. Pixi allows us to make this process easy. Running `pixi run setup` will automatically install all necessary packages, configure the relevant environment variables, and download the grizli configurations, including the latest NIRISS dispersion solutions and WFSS Backgrounds.

For advanced users who have existing CRDS cache and grizli conf locations, you can configure the defaults in `resources/setupShell.sh`. By default, these will be downloaded to extra directories within this directory. 


### Set Up Pipeline
This pipeline processes data in `fields` which are computed in the following way:
- If two NIRISS fields overlap (computed using a Cartesian approximation) they are added to the same field.
- If two NIRISS fields are within 30" of each other (computed using a spherical prescription), they are added to the same field. This is to account for the fact that the trace from a galaxy in one NIRISS field may fall into a non-contiguous field some distance away.
- Fields are named based on the constellation they are in along with a numeric index that indicates the relative start of observation of the field (e.g. the first data for field uma-00 arrived before uma-01).
To compute these fields we can run: `pixi run computeFields`
Anytime new data is taken it may be necessary to 

If you are running this on a cluster (or even on a powerful server) you will likely need to write a Snakemake configuration profile for the workflow. You can see examples of these in the `profiles/` directory. Please get in touch if you need one made for your environment.

Essentially this allows Snakemake to submit your SLURM jobs on your behalf. As such it will require basic SLURM information, such as the number of available CPUs per node and the number of jobs Snakemake can submit at once. 

In addition, if the packages necessary for Snakemake are updated, you will need to clean the Snakemake cache: `pixi run clean`

### Pipeline Description
The following is a brief description of Snakemake and the pipeline developed in this repository. All of the relevant scripts are contained within the `workflow` directory. Snakemake begins by reading the Snakefile which describes the rules necessary to create output files based on input files. Snakemake then computes the DAG necessary to create all requested output files and distributes them in parallel where possible. Each rule requires inputs, outputs, and code necessary to produce inputs from outputs. Here I briefly describe the stages of our pipeline:

0) Download Data. The UNCAL images for the field in question is downloaded. If the data exist, it will not re-download them. The download is parallelized.

1) Stage1 Processing. Here we convert UNCAL files for NIRISS into RATE files. Here we use the latest version of the JWST pipeline and implement the ColumnJump step which can offer improvements for NIRISS fields. In addition, we do some 1/f correction in Stage 1 for Imaging.

2) Preprocessing. Here the data is pre-processed using `grizli`. In a nutshell, a modified version of the Stage2 pipeline is applied, and dithered frames are drizzled together.

3) Mosaicing. Frames in the same field are mosaiced together and a detection catalog is measured on an IR image made from combining all available direct filters.

4) Contamination Modelling. Based on the direct image and detection catalog, we create a model of the dispersed image that can be used to model the contamination from overlapping traces. 

5) Spectral Extraction. 2D Spectra are extracted from the dispersed images. 

6) Redshift Fitting. 

7) FITSMap. In the final step, the products are collected and a FITSMap is created highlighting all of the results of the data analysis.

### Snakemake!
Now we just have to type one command and watch as our worflow is distributed over our cluster:
`pixi run snakemake --workflow-profile profiles/your-profile`
And that's it! Snakemake will distribute this workflow over the clusters for you! Two additional flags that you should be aware of:
- `--force-all`: Snakemake can recognize when it needs to re-run a rule in many cases, such as a change in the conda/mamba environment used for the rule, or a change in the Snakemake rule (see below). But it's always good to have a hammer when you want to force a full rerun.
- `--rerun-incomplete`: Sometimes a job/rule will fail, leaving behind bad files. Running with this option will get rid of pesky errors related to this, and force those jobs to rerun. 

If you just want to run a single field, simply provide the relevant field output file as input to the command: `snakemake FIELDS/leo-00/logs/fmap.log`.

If you've edited a step of the pipeline and want to force it to re-run just for that step, you can delete the relevant logfile from the log directory.

Note: For cluster users, you will need to make sure that the required environment modules are loaded in your bashrc/profile. There should be a way to do it with snakemake but I haven't managed to get it to work yet. Importantly, conda and TeX are needed for running this pipeline, which are available to the MPIA Cluster as modules.

Note: If you get tired of prepending `pixi run` each time, you can simply do `pixi shell` to launch a shell within the environment (much like `conda activate`).

### Summary
With just four commands we can process the entire pipeline:
```
git clone git@github.com:OutThere-JWST/NISPureParallel.git
pixi run setup
pixi run computeFields
pixi run snakemake
```
In practice, we likely want to customize the final step so as to activate a cluster profile if running over the entire dataset, or specify a specific field to process. 