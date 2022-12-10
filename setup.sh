#!/usr/bin/env
PROJ_HOME="$(pwd)"
export PROJ_HOME
export SCRIPTS_FOLDER="$PROJ_HOME/scripts"
export DATA_DIR="$PROJ_HOME/data"
export BIKES_FOLDER="$DATA_DIR/bikes"
export ENERGY_FOLDER="$DATA_DIR/energy"
export ROAD3D_FOLDER="$DATA_DIR/road3d"
export EXPERIMENT_OUTPUTS="$PROJ_HOME/Experiments"

mkdir -p "$DATA_DIR"
mkdir -p "$BIKES_FOLDER"
mkdir -p "$ENERGY_FOLDER"
mkdir -p "$ROAD3D_FOLDER"
mkdir -p "$EXPERIMENT_OUTPUTS"

bash scripts/getData.sh