#!/bin/bash

python src/train.py --model extratrees --normalize standardize --datamanip smote --fold 0 --grid grid
python src/train.py --model extratrees --normalize standardize --datamanip smote --fold 1 --grid grid
python src/train.py --model extratrees --normalize standardize --datamanip smote --fold 2 --grid grid
python src/train.py --model extratrees --normalize standardize --datamanip smote --fold 3 --grid grid
python src/train.py --model extratrees --normalize standardize --datamanip smote --fold 4 --grid grid

