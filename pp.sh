#!/bin/bash

usage() {
    cat <<EOM
    Usage:
    $(basename $0) -f {path to file}
EOM
    exit 0
}

[ -z "$1" ] && { usage; }

echo "$1"
echo "$2"
DIR="./DATA/split_files"
if [ -f "$2" ]; then
    LINES=$(zcat "$2" | wc -l)
    echo "$LINES"
    read -p "How many files would you like to split? " VAR
    CLC=$(expr "$LINES" / "$VAR")
    echo "$CLC"
#    # Check if directory does not exist
    if [ ! -d "$DIR" ]; then
        echo "Directory $DIR does not exist. Creating it now..."
        mkdir -p "$DIR"
        echo "Directory $DIR created successfully."
    else
        echo "Directory $DIR already exists."
    fi
    zcat "$2" | split -l "$CLC" - ./DATA/split_files/
    gzip DATA/split_files/a*
    zcat $2 | head -1 > DATA/output.json
    gzip DATA/output.json
fi
exit
