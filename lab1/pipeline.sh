#!/bin/bash

mkdir -p test train

python3 data_creation.py
python3 model_preprocessing.py
python3 model_preparation.py
python3 model_testing.py