#! /bin/bash
# Script to move dataset into RAM on compute node,
# TODO Update Data Set Location
echo "Running for dataset: " $1
rsync --info=progress2 /scratch/gjd9961/data/$1.squashfs /dev/shm/
