import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import config
from dispatchers import normalize_dispatcher

df = pd.read_csv(config.TRAINING_FILES[0])

df_train = df.drop("kfold", axis=1)
df_valid = df.drop("kfold", axis=1)

x_train = df_train.drop("target", axis=1).values
y_train = df_train.target.values

scaler = normalize_dispatcher.normalize["standardize"]
x_train = scaler.fit_transform(x_train)

pca = PCA(n_components=4)
principal_components = pca.fit_transform(x_train)
print(principal_components.shape)
y_train = y_train.reshape(-1, 1)
principal_components = np.hstack((principal_components, y_train))

# principal_components.join(y_train)
principalDf = pd.DataFrame(data=principal_components
             , columns = ['pc1', 'pc2', 'pc3', 'pc4', 'target'])

x = principalDf['pc1']
y = principalDf['pc2']
z = principalDf['pc3']
w = principalDf['pc4']
col = principalDf['target']
fig=plt.figure(figsize=(12,12))



axes = plt.axes(projection='3d')
axes.set_title("PCA: Dimensionality Reduction")
plt.suptitle("Customer Churn")
axes.set_xlabel("principal component 1")
axes.set_ylabel("principal component 2")
axes.set_zlabel("principal component 3")
scatter = axes.scatter(x, y, z, c=col, s=w, cmap='coolwarm')
axes.legend(*scatter.legend_elements())

plt.show()