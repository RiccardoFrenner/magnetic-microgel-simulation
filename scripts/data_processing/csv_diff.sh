#!/bin/bash

# Usage: find ~/data/mmgel/production4/ -iname '*bead*.npy' | ./scripts/data_processing/compute_all_hydro_radii.sh data/processed/radii_prod4.csv


# Check if the CSV file path is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <csv-file-path>"
  exit 1
fi

CSV_FILE=$1

# Temporary files for the sorted lists
LIST1=$(mktemp)
LIST2=$(mktemp)

# Read the first list from stdin and sort it
sort > "$LIST1"

# Extract the second column from the CSV file, sort it, and save it to LIST2
cut -d, -f1 "$CSV_FILE" | sort > "$LIST2"

# Compute the difference
comm -23 "$LIST1" "$LIST2"

# Clean up temporary files
rm "$LIST1" "$LIST2"
