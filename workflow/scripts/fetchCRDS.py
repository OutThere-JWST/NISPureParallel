#! /usr/bin/env python

from crds import getreferences
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table, vstack

# Concatenate all files
fields = vstack([Table(h.data) for h in fits.open('FIELDS/fields.fits')[1:]])

# Assign correct names to columns
fields['date-obs'] = [i.split('T')[0] for i in fields['date_obs']]
fields['time-obs'] = [i.split('T')[1] for i in fields['date_obs']]
fields['pupil'] = fields['niriss_pupil']

# Necessary keys for CRDS
keys = [
    'date-obs',
    'time-obs',
    'instrume',
    'detector',
    'filter',
    'pupil',
    'exp_type',
    'readpatt',
    'subarray',
]

# Downlaod the references
print('Downloading NIRISS references from CRDS...')
for f in tqdm(fields):
    getreferences({c.upper(): f[c] for c in keys})
