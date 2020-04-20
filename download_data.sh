#!/bin/bash

mkdir data
cd data

# Download own models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1lsDX8uUUWN8i2OnIp02kdUWhreSANMJ3' -O data.zip
unzip data.zip


# Download dataset
wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip -O dataset.zip
unzip dataset.zip
