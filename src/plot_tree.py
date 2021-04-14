import joblib
from sklearn import tree
import matplotlib.pyplot as plt

import config

if __name__ == "__main__":
    clf = joblib.load(config.INFERENCE_MODEL)
    plt.figure(figsize=(16,16))

    tree.plot_tree(clf, fontsize=10)
    plt.savefig("test.pdf")
