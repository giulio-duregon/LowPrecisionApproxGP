#! /bin/bash
# Script to move dataset into RAM on compute node,
# TODO Update Data Set Location
echo "Running for dataset: " $1
rsync --info=progress2 $DATA_DIR/$1/* /dev/shm/
