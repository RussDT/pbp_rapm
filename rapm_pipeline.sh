#!/bin/bash

# Exit immediately if any command fails
set -e

# Run the R script
echo "Running pbp_scrape2.R..."
Rscript pbp_scrape2.R

# Run the Python script
echo "Running process_rapm2.py..."
python3 process_rapm2.py

echo "Scripts completed successfully."
