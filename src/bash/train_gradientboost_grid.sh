#!/bin/bash

python src/train.py --model gradientboost --normalize standardize --datamanip smote --fold 0 --grid grid --subfeatures 1
python src/train.py --model gradientboost --normalize standardize --datamanip smote --fold 1 --grid grid --subfeatures 1
python src/train.py --model gradientboost --normalize standardize --datamanip smote --fold 2 --grid grid --subfeatures 1
python src/train.py --model gradientboost --normalize standardize --datamanip smote --fold 3 --grid grid --subfeatures 1
python src/train.py --model gradientboost --normalize standardize --datamanip smote --fold 4 --grid grid --subfeatures 1

python src/train.py --model adaboost --normalize standardize --datamanip smote --fold 0 --grid grid --subfeatures 1
python src/train.py --model adaboost --normalize standardize --datamanip smote --fold 1 --grid grid --subfeatures 1
python src/train.py --model adaboost --normalize standardize --datamanip smote --fold 2 --grid grid --subfeatures 1
python src/train.py --model adaboost --normalize standardize --datamanip smote --fold 3 --grid grid --subfeatures 1
python src/train.py --model adaboost --normalize standardize --datamanip smote --fold 4 --grid grid --subfeatures 1