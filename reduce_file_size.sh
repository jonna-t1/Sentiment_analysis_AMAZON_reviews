#!/bin/bash

# Script to safely extract the first N MB of JSON lines into a .json.gz file.

# Usage function
usage() {
    echo "Usage: $0 <input_file.gz> <target_no_reviews> <output_file.json.gz>"
    exit 1
}

# Ensure correct number of arguments
if [ "$#" -ne 3 ]; then
    usage
fi

# Input variables
input_file="$1"
target_size_mb="$2"
output_file="$3"

# Verify the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file $input_file does not exist."
    exit 1
fi

# Temporary files
temp_decompressed="temp_decompressed.json"
temp_sliced="temp_sliced.json"

# Decompress JSON data stream
echo "Decompressing the JSON stream..."
gunzip -c "$input_file" > "$temp_decompressed"

# Extract valid JSON lines until the size reaches the target MB
echo "Extracting JSON lines..."
head -n $2 "$temp_decompressed" > "$temp_sliced"

# Compress the JSON lines safely
gzip -c "$temp_sliced" > "$output_file"

# Cleanup temporary files
rm -f "$temp_decompressed" "$temp_sliced"

echo "Done! Data written to: $output_file"
