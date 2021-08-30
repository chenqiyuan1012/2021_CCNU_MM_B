import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
X = pd.read_csv('./data/bdtnp.csv')
X = X.drop('Unnamed: 0', axis=1)
corr_origin = X.corr()
temp = pd.read_csv('./data/bdtnp.csv')
temp = temp.drop('Unnamed: 0', axis=1)
boundary = 0
min_distance = 0
distances = []
for i in np.arange(0.25, 0.35, 0.01):
    res = pd.DataFrame()
    for j in range(84):
        slice = temp.iloc[:, j]
        slice = slice.mask(slice >= i, 1)
        slice = slice.mask(slice < i, 0)
        res = pd.concat([res, slice], axis=1, ignore_index=True)
    corr_temp = res.corr(method='pearson')
    corr_temp.fillna(0, inplace=True)
    dis = np.linalg.norm(np.array(corr_temp)-np.array(corr_origin), ord=2)
    print(dis)
    distances.append(dis)
    if(dis < min_distance or min_distance == 0):
        boundary = i
        min_distance = dis
        print('边界是', boundary)


x = np.arange(0.25, 0.35, 0.01)
y = distances
plt.plot(x, y)
plt.xlabel('boundary')
plt.ylabel('distance')
plt.show()
