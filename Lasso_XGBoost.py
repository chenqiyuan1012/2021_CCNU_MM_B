import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from skfeature.function.sparse_learning_based import MCFS, NDFS
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import heapq


def feature_selection():
    X = pd.read_csv('./data/bdtnp.csv')
    X = X.drop('Unnamed: 0', axis=1)
    idx = X.columns.values.tolist()
    y = pd.read_csv('./data/geometry.csv')
    y = y.drop('Unnamed: 0', axis=1)
    #y = y['xcoord']
    rows, columns = X.shape
    # 按照7:3切分训练集与测试集
    boundary = int(rows*0.7)
    X_train = X[:boundary]
    X_test = X[boundary:]
    y_train = y[:boundary]
    y_test = y[boundary:]
    reg = linear_model.MultiTaskLassoCV()
    #reg = linear_model.LassoCV()
    reg.fit(X_train, y_train)
    print(reg.alpha_)
    y_predict = reg.predict(X_test)
    print("mean_square_error:%.2f" % mean_squared_error(y_test, y_predict))
    coef = abs(reg.coef_)
    print(coef)
    x, y, z = coef[0], coef[1], coef[2]
    total = abs(x+y+z)
    total = list(total)
    ls = list(map(total.index, heapq.nlargest(20, total)))
    for l in ls:
        print(idx[l])
    coef = pd.Series(total, index=X_train.columns)
    return np.array(pd.concat([coef.sort_values().head(20)]))


x = pd.read_csv('data/binarized_bdtnp.csv')
y = pd.read_csv('data/geometry.txt', sep=' ')
print(x, y)
idx = feature_selection(20)
x = x.loc[:, idx]
print(x.shape)

MOR = MultiOutputRegressor(XGBRegressor()).fit(x, y)

x_pred = pd.read_csv('data/dge_binarized_distMap.csv')
x_pred = pd.DataFrame(
    x_pred.values.T, index=x_pred.columns, columns=x_pred.index)
x_pred = x_pred.loc[:, idx]
y_pred = MOR.predict(x_pred)
print(y_pred.shape)
y_pred = pd.DataFrame(y_pred)
print(y_pred)
indices = []
for i in range(1297):
    loc_pre = np.full((3039, 3), np.array(y_pred.iloc[i]))
    dist = scipy.spatial.distance.cdist(
        loc_pre, np.array(y))[0]
    top_idx = dist.argsort()[:10]
    top_idx = [e+1 for e in top_idx]
    print(top_idx)
    print(len(top_idx))
    indices += [top_idx]


dout = '20.csv'
with open(dout, 'w') as file_result:
    for i in range(len(indices)):
        tmp = [i+1] + indices[i]
        file_result.write(','.join(map(str, tmp)) + '\n')
