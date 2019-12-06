#!/bin/bash

# Base of the CAIDA telescope URL
base=data.caida.org/datasets/security/telescope-educational/exercises/anon/

# The file names to be retrieved from the CAIDA website
# The 10-digit numbers are UNIX time stamps for one hour
# within a day
readarray files < telescope_files.txt

for f in "${files[@]}"
do
    rm -f anon/$f
    wget -P anon ${base}$f
done
