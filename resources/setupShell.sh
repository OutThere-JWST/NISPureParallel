#! /usr/bin/env sh

# Overwrite if custom paths are needed
export GRIZLI="$(pwd)/GRIZLI"
export CRDS_PATH="$(pwd)/CRDS_CACHE"

# Export GRIZLI-related environment variables
export iref="$GRIZLI/iref"
export jref="$GRIZLI/jref"

# Export CRDS-related environment variables
export CRDS_CONTEXT="jwst_1314.pmap"
export CRDS_SERVER_URL="https://jwst-crds.stsci.edu"
export CRDS_READONLY_CACHE="1"