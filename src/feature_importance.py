import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

df = pd.read_csv(config.TRAINING_FILES[0])
clf = joblib.load(config.INFERENCE_MODEL)

importances = clf.feature_importances_
features = df.columns
sorted_indices = np.argsort(importances)

plt.title('Feature Importances')
plt.xlabel('Relative Importance')

plt.barh(range(len(sorted_indices)), importances[sorted_indices], color='b', align='center')
plt.yticks(range(len(sorted_indices)),
           [features[i] for i in sorted_indices],
           fontsize=7,
           rotation=30)


plt.tight_layout()
plt.show()
