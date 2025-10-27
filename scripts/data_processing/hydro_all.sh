LIST1=$(mktemp)

find ~/data/mmgel/production4/ -iname '*bead*.npy' | ./scripts/data_processing/csv_diff.sh data/processed/radii_prod4.csv > "$LIST1"

# Count the number of files to be processed
NUM_FILES=$(wc -l < "$LIST1")
echo "Processing $NUM_FILES files"

cat "$LIST1" | parallel --bar -j 32 ./scripts/data_processing/hydro_radius {} >> data/processed/radii_prod4.csv 