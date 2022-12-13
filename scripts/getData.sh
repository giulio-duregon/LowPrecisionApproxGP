#! /bin/bash

# Get Bikes Dataset, unzip folder, remove old zip
wget -r -O "$BIKES_FOLDER"/bikes https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
unzip "$BIKES_FOLDER"/bikes -d "$BIKES_FOLDER"
rm "$BIKES_FOLDER"/bikes

# # Get 3droad dataset
wget -O "$ROAD3D_FOLDER"/3droad.txt https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt

# Get Energy Dataset, Convert From Excel -> CSV Energy sorta fd us
wget -O $ENERGY_FOLDER/energy.xlsx https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx
python "$SCRIPTS_FOLDER"/xlsx2csv.py -i "$ENERGY_FOLDER"/energy.xlsx -o "$ENERGY_FOLDER"/energy.csv
rm "$ENERGY_FOLDER"/energy.xlsx

wget -O "$NAVAL_FOLDER"/naval https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip
unzip "$NAVAL_FOLDER"/naval -d "$NAVAL_FOLDER"
rm "$NAVAL_FOLDER"/naval

wget -O "$PROTEIN_FOLDER"/casp.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv
