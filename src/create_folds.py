import os
import sys
import json
import argparse
from sklearn import model_selection, preprocessing
import pandas as pd
import config
import files
import numpy as np

def run(folds=5, force_rewrite=False):
    cwd = os.path.dirname(os.path.realpath(__file__))      #path to current file
    parent_dir = os.path.split(cwd)[0]

    dataset = pd.read_csv(config.KFOLD_INPUT)

    # Cleanup Data
    del dataset["customerID"]

    # label_encoder = preprocessing.LabelEncoder()
    to_categorical = [
        "InternetService",
        "Contract",
        "PaymentMethod",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines"
    ]

    for field in to_categorical:
        print(f"Encoding {field}")

        ohe = preprocessing.OneHotEncoder(sparse=False)
        one_hot_values = ohe.fit_transform(dataset[field].values.reshape(-1,1))

        for x in range(one_hot_values.shape[1]):
            column_name = f'{field}_{ohe.categories_[0][x]}'.replace(' ', '')
            dataset[column_name] = one_hot_values[:, x].astype(int)

        # drop the old column
        del dataset[field]


    to_bool = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "target"
    ]

    for field in to_bool:
        print(f"Encoding {field}")

        if field == "gender":
            dataset["IsFemale"] = dataset[field].map(
                {'Female':1 ,'Male':0})
            del dataset["gender"]

        else:
            dataset[field] = dataset[field].map(
                {'Yes':1 ,'No':0})


    to_float_64 = [
        "TotalCharges"
    ]

    for field in to_float_64:
        dataset[field] = pd.to_numeric(dataset[field], errors='coerce')
        dataset[field] = dataset[field].replace(np.nan, 0)

    print(dataset)

    print('creating KFolds')
    dataset["kfold"] = -1

    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset.target.values

    kf = model_selection.StratifiedKFold(n_splits=folds)

    for f, (t_, v_) in enumerate(kf.split(X=dataset, y=y)):
        dataset.loc[v_, 'kfold'] = f

    dirs = config.KFOLD_INPUT.replace('raw', 'processed').split('/')
    new_parent_dir = parent_dir
    for dir in dirs[:-1]:
        new_parent_dir = os.path.join(new_parent_dir, dir)
        if not os.path.exists(new_parent_dir):
            print(f"+w {new_parent_dir}")
            os.mkdir(new_parent_dir)

    csv_path = os.path.join(new_parent_dir, f"customer_churn.csv").replace('raw', 'processed')

    for field in dataset.columns:

        val = dataset[field].isnull().values.any()
        if val:
            print(field)
    if not os.path.exists(csv_path) or force_rewrite:
        print(f"+w {csv_path}")
        dataset.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--force", type=bool)
    parser.add_argument("--folds", type=int)


    args = parser.parse_args()

    run(
        folds=args.folds,
        force_rewrite=args.force
        )