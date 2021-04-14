import matplotlib.pyplot as plt
import pandas as pd

from dispatchers import normalize_dispatcher
import config
"""
Code derived from: https://stackoverflow.com/questions/55756555/how-to-combine-2-dataframe-histograms-in-1-plot

"""

for file in config.TRAINING_FILES:
    df = pd.read_csv(file)
    df = df.drop("target", axis=1)
    df = df.drop("kfold", axis=1)
    labels = df.columns
    scaler = normalize_dispatcher.normalize["standardize"]
    df = pd.DataFrame(scaler.fit_transform(df), columns=labels)
    fig, axes = plt.subplots(8,5, figsize=(16,10))
    fig.suptitle("Customer Churn Feature Distribution")
    df.plot(kind='hist', subplots=True, ax=axes, alpha=0.5, fontsize=4)

    # clone axes so they have different scales
    ax_new = [ax.twinx() for ax in axes.flatten()]
    df.plot(kind='kde', ax=ax_new, subplots=True)

    plt.show()

