#! /usr/bin/env python

# Import packages
import os
import sys
import glob
import shutil
import argparse
import warnings
import numpy as np
from matplotlib import pyplot
from astropy.io import fits
from astropy.table import Table
from astropy.io.fits import getheader,getdata
from astropy.coordinates import Angle

# Silence warnings
warnings.filterwarnings('ignore')

# Parse arguements
parser = argparse.ArgumentParser()
parser.add_argument('clusterid', type=str)
args = parser.parse_args()
cname = args.clusterid

# Get paths and get clusters
main = os.getcwd()
clusters = os.path.join(main,'CLUSTERS')
prods = Table.read(os.path.join(clusters,'cluster-prods.fits'),cname)
home = os.path.join(clusters,cname)
print(f'Preprocessing {cname}')

# Get main paths
rate = os.path.join(main,'RATE')
if not os.path.exists(home): os.mkdir(home)

# Subdirectories
raw = os.path.join(home,'RAW')
logs = os.path.join(home,'logs')
prep = os.path.join(home,'Prep')
plots = os.path.join(home,'Plots')
persist = os.path.join(home,'Persistence')
extract = os.path.join(home,'Extractions')
for d in [raw,logs,prep,plots,persist,extract]:
    if os.path.exists(d): shutil.rmtree(d)
    os.mkdir(d)

# Redirect stdout and stderr to file
sys.stdout = open(os.path.join(logs,'proc.out'),'w')
sys.stderr = open(os.path.join(logs,'proc.err'),'w')

# Change to working directory
os.chdir(clusters)

# Import grizli
import jwst,grizli
from grizli import utils
from grizli import jwst_utils
from grizli.aws import visit_processor
from grizli.pipeline import auto_script

# Print grizli and jwst versions
print(f'grizli:{grizli.__version__} | jwst:{jwst.__version__}')

# Initialize image
files = [os.path.join(rate,f).replace('uncal','rate') for f in prods['productFilename']]
for f in files:

    # Copy raw file
    f = shutil.copy(f,raw)

    # Initialize image
    jwst_utils.initialize_jwst_image(f)
    jwst_utils.set_jwst_to_hst_keywords(f,reset=True)

# Parse Visits
visits, all_groups, info = auto_script.parse_visits(field_root=cname,RAW_PATH=raw)

# Get color cycle
colors = [c['color'] for c in pyplot.rcParams['axes.prop_cycle']]

# Create figure
fig, ax = pyplot.subplots(1,1,figsize=(12,12))

# Enumerate visits
for i, v in enumerate(visits):

    # Get color
    c = colors[i%len(colors)]

    # Get region box
    sr = utils.SRegion(v['footprint'])

    # Plot center 
    ax.scatter(*sr.centroid[0], marker='.', c=c)

    # Place patches for region
    for patch in sr.patch(ec=c, fc='None', alpha=0.5, label=v['product'],lw=3,ls='--'): ax.add_patch(patch)

# Set axis parameters
ax.set_aspect(1./np.cos(ax.get_ylim()[0]/180*np.pi)) # square with cos(dec)
ax.set_xlim(ax.get_xlim()[::-1]) # Reverse Xlim
ax.legend(fontsize=10,frameon=True)
ax.grid(ls=':')

# Format labels correctly
ax.set_xticklabels([Angle(t,unit='deg').to_string(unit='hour', sep=[r'$^\textrm{'+s+'}$' for s in 'hms']) for t in ax.get_xticks()], rotation=25)
ax.set_yticklabels([Angle(t,unit='deg').to_string(unit='degree', sep=(r'$^\circ$',r"$'$",r"$''$")) for t in ax.get_yticks()])
ax.set(xlabel='Right Ascension (ICRS)',ylabel='Declination (ICRS)')

# Save figure
fig.savefig(os.path.join(plots,'regions.pdf'),bbox_inches='tight')
pyplot.close(fig)

# Make visit associations
assoc = info['EXPSTART','EXPTIME','INSTRUME']
assoc['filter'] = ['-'.join(f.split('-')[:0:-1]) for f in info['FILTER']]

# Find footprints
footprints = []
for f in sorted(glob.glob(f'{raw}/*rate.fits')):
    for v in visits:
        if os.path.basename(f) in v['files']:
            footprints.append(v['footprints'][v['files'].index(os.path.basename(f))])
            break

# Fill out association table
assoc['t_max'] = [int(getheader(f,1)['MJD-END']) for f in files]
assoc['proposal_id'] = [int(getheader(f,0)['PROGRAM']) for f in files]
assoc['footprint'] = footprints
assoc['dataURL'] = [f"mast:JWST/product/{os.path.split(f)[1].replace('uncal','rate')}" for f in files]

# Rename columns
assoc.rename_columns(['EXPSTART','EXPTIME','INSTRUME'],['t_min','exptime','instrument_name'])

# Visit process arguements
prep_args = {
    'tweak_max_dist':5,
    'oneoverf_kwargs':None,
    'snowball_kwargs':None
}

# Other arguements
other_args = {
    'is_parallel_field':True,
}

# Process visit
visit_processor.process_visit(cname,tab=assoc,prep_args=prep_args,other_args=other_args,clean=False, sync=False,with_db=False,visit_split_shift=2,combine_same_pa=True)

# Plot Science Images
files = glob.glob(os.path.join(prep,'*drz_sci.fits'))
for file in files:

    # Split path
    f = os.path.split(file)[1]

    # Create figure 
    fig, ax = pyplot.subplots(1,1,figsize=(8,8))

    # Get image data
    data = getdata(file)

    # Scaling
    vm = np.nanpercentile(data, [5, 95])

    # Plot images
    ax.imshow(data, vmin=-0.1*vm[1], vmax=vm[1], cmap='gray_r')

    # Set axis
    ax.set_title(f)
    ax.axis('off')
    ax.set_aspect(1)

    # Save and close
    fig.savefig(os.path.join(plots,f.replace('.fits','.pdf')),bbox_inches='tight')
    pyplot.close(fig)