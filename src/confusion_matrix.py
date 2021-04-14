# Taken directly from sklearn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import config
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def run(fold):
    for file in config.TRAINING_FILES:
        df = pd.read_csv(file)
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        df_train = df_train.drop("kfold", axis=1)
        df_valid = df_valid.drop("kfold", axis=1)

        x_train = df_train.drop("target", axis=1).values
        y_train = df_train.target.values

        x_valid = df_valid.drop("target", axis=1).values
        y_valid = df_valid.target.values


        clf = joblib.load(config.INFERENCE_MODEL)


        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [(f"Confusion matrix, without normalization: Fold {fold}", None),
                        (f"Normalized confusion matrix: Fold {fold}", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(clf, x_valid, y_valid,
                                        display_labels=["NoChurn", "Churn"],
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
            disp.ax_.set_title(title)
            plt.suptitle(config.INFERENCE_MODEL)

            print(title)
            print(disp.confusion_matrix)

        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    args = parser.parse_args()

    run(fold=args.fold)