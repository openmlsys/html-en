#!/bin/bash

set -e

rm -rf _build/rst _build/html
d2lbook build rst
cp static/frontpage.html _build/rst/
d2lbook build html
cp -r static/image _build/html/_static
cp -r static/* _build/html/_static
python3 tools/format_tables.py
touch _build/html/.nojekyll

cp -r _build/html _build/docs
