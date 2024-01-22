#!/bin/bash

# Input JSON file
json_file="Books.json"

# Output text file
output_file="data.txt"

# Extract reviewText from the first 10 JSON objects and concatenate them
jq -r '.reviewText' "$json_file" | head -n 1000000 > "$output_file"

echo "Subset concatenation complete. Output saved to $output_file"
