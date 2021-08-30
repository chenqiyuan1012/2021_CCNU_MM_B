import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import heapq
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
imp_coef = np.array(pd.concat([coef.sort_values().head(20)]))
