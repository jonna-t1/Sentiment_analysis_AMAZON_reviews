#!/bin/bash

usage() {
	    cat <<EOM
    Usage:
    $(basename $0) -f {path to file}

EOM
    exit 0
}

[ -z $1 ] && { usage; }

echo $1
echo $2
if [ -f $2 ]; then
LINES=`zcat $2 | wc -l`
echo $LINES
read -p "How many files would you like to split?" VAR
CLC=$(expr $LINES / $VAR)
echo $CLC
zcat $2 | split -l $CLC - ./DATA/
gzip DATA/a*
zcat DATA/Electronics_5.json.gz | head -1 > DATA/output.json
fi;
exit;
