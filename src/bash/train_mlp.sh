#!/bin/bash

python src/train.py --model mlp-opt --normalize standardize --datamanip smote --fold 0
python src/train.py --model mlp-opt --normalize standardize --datamanip smote --fold 1
python src/train.py --model mlp-opt --normalize standardize --datamanip smote --fold 2
python src/train.py --model mlp-opt --normalize standardize --datamanip smote --fold 3
python src/train.py --model mlp-opt --normalize standardize --datamanip smote --fold 4