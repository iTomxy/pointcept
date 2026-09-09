#!/bin/sh
set -eu

# Refresh the upstream class map in this package. Derived constants belong in
# classes.py so this vendored file can always be replaced verbatim.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmp_map=$(mktemp "$script_dir/map_to_binary.py.tmp.XXXXXX")
trap 'rm -f "$tmp_map"' EXIT HUP INT TERM
curl -fL \
  https://raw.githubusercontent.com/wasserth/TotalSegmentator/master/totalsegmentator/map_to_binary.py \
  -o "$tmp_map"
mv "$tmp_map" "$script_dir/map_to_binary.py"
trap - EXIT HUP INT TERM
