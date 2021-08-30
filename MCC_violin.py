import scipy
import numpy as np
import pandas as pd
from skfeature.function.sparse_learning_based import MCFS, NDFS
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import heapq


def feature_selection(num):
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


def get_mcc(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    mcc = (TP * TN) - (FP * FN)
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denom == 0:
        return 0
    return mcc / denom


def xgb():
    idx = feature_selection(60)
    origin = pd.read_csv('data/binarized_bdtnp.csv')
    target = pd.read_csv('data/dge_binarized_distMap.csv')
    result = pd.read_csv('XGB_result/60.csv')
    target = pd.DataFrame(
        target.values.T, index=target.columns, columns=target.index)
    origin = origin.loc[:, idx]
    target = target.loc[:, idx]
    avg = []
    for i in range(1297):
        slice = np.array(result.iloc[i])  # 包含10个位置
        tg = np.array(target.iloc[i])  # 预测位置的60维向量
        dis = 0
        for j in range(10):
            ori = np.array(origin.iloc[slice[j]-1])
            dis += get_mcc(tg, ori)
        avg.append(dis/10)
        print(len(avg))
    return avg


def knn():
    idx = feature_selection(60)
    origin = pd.read_csv('data/binarized_bdtnp.csv')
    target = pd.read_csv('data/dge_binarized_distMap.csv')
    result = pd.read_csv('KNN_result/60.csv')
    target = pd.DataFrame(
        target.values.T, index=target.columns, columns=target.index)
    origin = origin.loc[:, idx]
    target = target.loc[:, idx]
    avg = []
    for i in range(1297):
        slice = np.array(result.iloc[i])  # 包含10个位置
        tg = np.array(target.iloc[i])  # 预测位置的60维向量
        dis = 0
        for j in range(10):
            ori = np.array(origin.iloc[slice[j]-1])
            dis += get_mcc(tg, ori)
        avg.append(dis/10)
        print(len(avg))
    return avg


def mcc():
    idx = feature_selection(60)
    origin = pd.read_csv('data/binarized_bdtnp.csv')
    target = pd.read_csv('data/dge_binarized_distMap.csv')
    result = pd.read_csv('MCC_result/60.csv')
    target = pd.DataFrame(
        target.values.T, index=target.columns, columns=target.index)
    origin = origin.loc[:, idx]
    target = target.loc[:, idx]
    avg = []
    for i in range(1297):
        slice = np.array(result.iloc[i])  # 包含10个位置
        tg = np.array(target.iloc[i])  # 预测位置的60维向量
        dis = 0
        for j in range(10):
            ori = np.array(origin.iloc[slice[j]-1])
            dis += get_mcc(tg, ori)
        avg.append(dis/10)
        print(len(avg))
    return avg


avg_xgb = xgb()
avg_knn = knn()
avg_mcc = mcc()
print('ok')
plt.violinplot((avg_knn, avg_mcc, avg_xgb))
plt.ylabel("Mean MCC")
plt.title("60 Genes")
plt.show()
print('okok')
