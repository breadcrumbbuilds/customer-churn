# Machine Learning Final Project: Customer Churn

Thompson Rivers University

COMP 4980 - Special Topics, Machine Learning

Bradley Crump

This project aims to show a student how to interact with data, formalize a plan for creating a predictive model and implement that model.

### Repository Information

Much of the project structure derives from Abishek Thakur's book, [Approaching Almost Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost).

The repo's code can be interacted with through the command line. The goal is to create an append-only, reusable codebase that can be applied to arbitrary use-cases. However, there are dependencies.

## Dependencies

There is a requirements.txt file that to the best of my knowledge contains the requirements for training in this repository. The packages of note include:
- Scikit-learn
- imbalanced-learn
- joblib
- matplotlib
- numpy

## Training

A note: Run scripts from the root of the project


This repo supports training a model from the command line at the root of the project. There is also funcitonality to execute several scripts. Examples of these scripts can be found in the [src/bash](src/bash) folder
Training a model comes with several options. The general command structure for training a model:

```python
python src/train.py \
    --model <a model (key) from src/dispatchers/model_dispatcher> \
    --fold <fold number (0 - 4 by default)> \
    --datamanip <a manip (key) from src/dispatchers/data_manip_dispatcher> \
    --grid <a grid (key) from src/dispatcher/grid_dispatcher> \
    --normalize <a noramlize (key) from src/dispatcher/normalize_dispatcher> \
    --subfeatures <defined in config as FEATURES>
```

A functional example:

```python
python src/train.py \
    --model rf \
    --fold 0 \
    --datamanip smote \
    --grid grid \
    --normalize standardize \
    --subfeatures 1
```

- `model` is a scikit learn model
- `fold` is the stratified k fold that will be treated as the validation set

The following commands are optional

- `datamanip` is the sampling strategy used
- `grid` which grid search to use. Will use the config.GRID_SEARCH_PARAMETERS
- `normalize` is used to scale the data
- `subfeatures` provides functionality for training a model on a subset of the features. Will use the config.FEATURES. If this command isn't used, all the features in the csv file in TRAINING_FILE will be used

The resulting model will be persisted to the /models folder. You can use the path to the model in the config.INFERENCE_MODEL to do other cool things in this repo like create a confusion matrix!

## Dispatchers

The repo uses a 'dispatching' framework that supports dynamic instantiation of different third party classes. The goal is to create an append only file that contains arbitrary complexity, however, there are drawbacks to this strategy, namely, the duplication of objects.

Currently, there are 4 dispatchers implemented.

- grid_dispatcher: dispatches different Grid Searching mechanisms from Sklearn
- data_manip_dispatcher: disptaches different sampling strategies from imblearn
- model_dispatcher: dispatches different Sklearn models
- normalize_dispatcher: dispatches scaling strategies from Sklearn

### Analysis

`Confusion Matrix`
Use confusion_matrix.py to produce a visualization of a confusion matrix.

Config Dependencies:

- INFERENCE_MODEL - model that will be used to create confusion matrix
- TRAINING_FILES - First file in the list will be used to retrieve the dataset

```python
python src/conufusion_matrix.py --fold <fold-number>
```

`Feature Importances`
Use feature_importance.py to produce a visualizaiton of the importance of features.

Config Dependencies:
- INFERENCE_MODEL - model that will be used to create confusion matrix
- TRAINING_FILES - First file in the list will be used to retrieve the dataset

```python
python src/feature_importance.py
```
