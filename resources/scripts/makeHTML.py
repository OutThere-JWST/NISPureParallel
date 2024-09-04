#! /usr/bin/env python

# Import packages
import os
import yaml
import numpy as np

# Astropy packages
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord

# Geometry packages
from sregion import SRegion
from shapely import union_all
from spherical_geometry.polygon import SingleSphericalPolygon


# Convert Shapely Polygon to Spherical
def get_area(reg):
    # Check if MultiPolygon
    if hasattr(reg, 'geoms'):
        return sum([get_area(g) for g in reg.geoms])

    # Get SingleSphericalPolygons
    xy = reg.exterior.xy
    return SingleSphericalPolygon.from_radec(*xy, center=np.mean(xy, 1)).area()


# Alias of fields
with open('resources/aliases.yaml', 'r') as file:
    aliases = yaml.safe_load(file)

# Open products file
hdul = fits.open('FIELDS/field-obs.fits')

# Iterate over fields
rows = []
for hdu in hdul[1:]:
    # Get field name
    fname = hdu.header['EXTNAME']
    obs = Table(hdu.data)

    # Get alias of field
    alias = aliases[fname] if fname in aliases.keys() else '\u200b'

    # Get link to field
    flink = f'<a href="s3/maps/{fname}/index.html" target="_blank">{fname}</a>'

    # Get field area
    reg = union_all([SRegion(o).shapely[0] for o in obs['s_region']])
    area = np.round(get_area(reg) * (60**4), 1)  # In arcsec^2

    # Get field obstime
    obs_time = round((obs['t_exptime']).sum())
    h, m = obs_time // 3600, obs_time % 3600 // 60
    exptime = f'{str(h).zfill(2)}h{str(m).zfill(2)}m'

    # Get field coordinates
    coord = SkyCoord(*reg.centroid.coords.xy, unit='deg')[0]
    ra, dec = [c[:-3] for c in coord.to_string('hmsdms', precision=0).split()]
    ra = ra.replace('h', '<sup>h</sup>').replace('m', '<sup>m</sup>')
    dec = dec.replace('d', '&deg;').replace('m', "'")

    # Get the filters
    gfs = np.unique(obs['filters'])
    gfs = {
        f: [gf.split(';')[0] for gf in gfs if f in gf]
        for f in np.unique([gf.split(';')[-1] for gf in gfs])
    }
    gfs = (
        ', '.join([f'{k} ({", ".join(v)})' for k, v in gfs.items()])
        .replace('CLEAR', 'D')
        .replace('GR150', '')
    )

    # Get started and end times
    t_start = Time(obs['t_obs_release'].min(), format='mjd').iso.split()[0]
    t_end = Time(obs['t_obs_release'].max(), format='mjd').iso.split()[0]

    # JWST IDs
    prop_ids = ', '.join(np.unique([o[3:7] for o in obs['obs_id']]))
    prim_ids = np.unique(obs['prim_id'])
    jwst_url = 'https://www.stsci.edu/jwst/science-execution/program-information?id='
    prim_ids = ', '.join(
        [f'<a href="{jwst_url}{i}" target="_blank">{i}</a>' for i in prim_ids]
    )

    # Get row
    rows.append(
        [flink, alias, ra, dec, area, exptime, gfs, t_start, t_end, prop_ids, prim_ids]
    )

# Transpose list of lists
names = [
    'Field',
    'Alias',
    'RA',
    'Dec',
    'Area',
    'Exptime',
    'Filters (Direct, Col, Row)',
    'Start',
    'End',
    'PIDs',
    'Parallel',
]
tab = Table(list(map(list, zip(*rows))), names=names)

# Save table to (HTML)
tab.write('HTML.html', format='html', overwrite=True)

# Read and delete the HTML file
with open('HTML.html', 'r') as f:
    table = ''.join(f.readlines()[6:-3])
os.remove('HTML.html')

# Replace relevant data
table = (
    table.replace('&amp;', '&')
    .replace('&gt;', '>')
    .replace('&lt;', '<')
    .replace('<table>', '<table id="data" class="display">')
)

# Read and replace the table in the HTML template
with open('resources/html5up-phantom/table-blank.html', 'r') as f:
    html = f.read()
    html = html.replace('<!-- TABLE HERE -->', table)
with open('resources/html5up-phantom/table.html', 'w') as f:
    f.write(html)

# Create Image HTML template
#
image_blank = """<article class="style0">
    <span class="image">
        <img src="s3/maps/{fname}/RGB/0/0/0.png" onerror="if (this.src != 'images/error.png') this.src = 'images/error.png';" />
    </span>
    <a href="s3/maps/{fname}/index.html" target="_blank">
        <h2>{fname}</h2>
        <div class="content">
            <p>{info}</p>
        </div>
    </a>
</article>"""

# Create Images HTML
images = []
for i, hdu in enumerate(hdul[1:]):
    # Get field name
    fname = hdu.header['EXTNAME']

    # Replace relevant data
    images.append(image_blank.format(fname=fname, info=f'PIDs: {tab[i]['PIDs']}'))
images = '\n'.join(images)

# Read and replace the images in the HTML template
with open('resources/html5up-phantom/images-blank.html', 'r') as f:
    html = f.read()
    html = html.replace('<!-- IMAGES HERE -->', images)
with open('resources/html5up-phantom/images.html', 'w') as f:
    f.write(html)
