from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model
from skfeature.function.sparse_learning_based import MCFS, NDFS
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import heapq


def method1():
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
    ls = list(map(total.index, heapq.nlargest(60, total)))
    for l in ls:
        print(idx[l])
    coef = pd.Series(total, index=X_train.columns)
    return np.array(pd.concat([coef.sort_values().head(60)]))


def method2():
    insitu_df = pd.read_csv('data/bdtnp.txt', sep='\t')
    marker_genes = list(insitu_df.columns)

    cell_df = pd.read_csv('data/dge_normalized.txt', sep='\t')
    all_genes = list(cell_df.index)
    cell_df = cell_df.T
    cell_df.columns = all_genes
    cell_df = cell_df[marker_genes]

    sel = VarianceThreshold()
    sel.fit(cell_df.values)
    score = sel.variances_

    # 对方差评分进行降序排序。将选择60个最高方差的基因。
    idx = np.argsort(score, 0)[::-1]
    genes = [marker_genes[e] for e in idx][:60]
    return genes


ge1 = method1()
ge2 = method2()
s1, s2 = set(ge1), set(ge2)
print(len(s1.intersection(s2)))
