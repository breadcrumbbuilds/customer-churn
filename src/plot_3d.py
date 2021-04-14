import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import config
from dispatchers import normalize_dispatcher

df = pd.read_csv(config.TRAINING_FILES[0])
df = df[config.FEATURES]

x_train = df.drop("target", axis=1).values
y_train = df.target.values

scaler = normalize_dispatcher.normalize["standardize"]
x_train = scaler.fit_transform(x_train)

y_train = y_train.reshape(-1, 1)
df = np.hstack((x_train, y_train))

# principal_components.join(y_train)
df = pd.DataFrame(data=df
             , columns = config.FEATURES)

x = df[config.FEATURES[0]]
y = df[config.FEATURES[1]]
z = df[config.FEATURES[2]]
if len(config.FEATURES) > 3:
    w = df[config.FEATURES[3]]
col = df['target']
fig=plt.figure(figsize=(12,12))



axes = plt.axes(projection='3d')
axes.set_title("Interesting Feature Analysis")
plt.suptitle("Customer Churn")
axes.set_xlabel(config.FEATURES[0])
axes.set_ylabel(config.FEATURES[1])
axes.set_zlabel(config.FEATURES[2])
if len(config.FEATURES) > 4:
    scatter = axes.scatter(x, y, z, c=col, s=w*w, cmap='coolwarm')
else:
    scatter = axes.scatter(x, y, z, c=col, cmap='coolwarm')
axes.legend(*scatter.legend_elements())
plt.tight_layout()
plt.show()