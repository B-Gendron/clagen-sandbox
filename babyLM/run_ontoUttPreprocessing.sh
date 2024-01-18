#!/bin/bash

# Exits if a command fails
set -e

# Reach the right location to clone OntoUttPreprocessing repository
cd ../..
git clone https://github.com/B-Gendron/OntoUttPreprocessing.git

# Go to the cloned repository
cd OntoUttPreprocessing

# Check if the parameter is provided
if [ -z "$1" ]; then
    echo "Please provide a parameter: 'data', 'individuals', or 'both'."
    exit 1
fi

# Execute the required code based on the provided parameter
cd code
if [ "$1" = "data" ] || [ "$1" = "both" ]; then
    echo Load and store data from wikipedia talk pages
    python3 preprocessing.py
fi

if [ "$1" = "individuals" ] || [ "$1" = "both" ]; then
    echo Create the ontology individuals
    python3 create_individuals.py
fi