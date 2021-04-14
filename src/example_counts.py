# Taken directly from sklearn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import config
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def run(feature):
    for file in config.TRAINING_FILES:
        df = pd.read_csv(file)
        if feature not in df.columns:
            print(f'{feature} not in columns of df')
            print(df.columns)
            return

        print(feature)
        print(df[feature].value_counts())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature",
        type=str
    )

    args = parser.parse_args()

    run(feature=args.feature)