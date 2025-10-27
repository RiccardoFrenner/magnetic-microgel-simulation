cat data/processed/gelEq_beads_latest.txt | parallel ~/software/espresso-4.2/build/pypresso scripts/data_processing/crosslink_rdf.py {}
