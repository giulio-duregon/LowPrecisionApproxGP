#! /bin/bash

# Get Bikes Dataset, unzip folder, remove old zip
wget -r -O "$BIKES_FOLDER"/bikes https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
unzip "$BIKES_FOLDER"/bikes -d "$BIKES_FOLDER"
rm "$BIKES_FOLDER"/bikes

# Get 3droad dataset
wget -O "$ROAD3D_FOLDER"/3droad.txt https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt

# Get Energy Dataset, Convert From Excel -> CSV
wget -O "$ENERGYFOLDER"/energy.xlsx https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx
python "$SCRIPTS_FOLDER"/xlsx2csv.py -i "$ENERGYFOLDER"/energy.xlsx -o "$ENERGYFOLDER"/energy.csv
rm "$ENERGYFOLDER"/energy.xlsx

wget -O "$NAVAL_FOLDER"/naval https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip
unzip "$NAVAL_FOLDER"/naval -d "$NAVAL_FOLDER"
rm "$NAVAL_FOLDER"/naval
