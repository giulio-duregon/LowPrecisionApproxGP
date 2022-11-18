#!bin/bash
export PROJ_HOME=$(pwd)
export SCRIPTS_FOLDER="$PROJ_HOME/scripts"
export DATA_DIR="$PROJ_HOME/data"
export BIKES_FOLDER="$DATA_DIR/bikes"
export ENERGYFOLDER="$DATA_DIR/energy"
export ROAD3D_FOLDER="$DATA_DIR/road3d"

mkdir -p $DATA_DIR
mkdir -p $BIKES_FOLDER
mkdir -p $ENERGYFOLDER
mkdir -p $ROAD3D_FOLDER
