import numpy as np
import pandas as pd
import scipy
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


def find_largest_MCC(genes):

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

    insitu_bin = pd.read_csv('data/binarized_bdtnp.csv')
    # # in-situ coordinates
    insitu_coords = pd.read_csv('data/geometry.txt', sep=' ')

    cell_bin = pd.read_csv('data/dge_binarized_distMap.csv')
    all_genes = list(cell_bin.index)
    cell_bin = cell_bin.T
    cell_bin.columns = all_genes

    Pr = cell_bin[genes].values
    Gt = insitu_bin[genes].values
    mcc = np.asarray([get_mcc(p, g) for p in Pr for g in Gt])
    mcc = mcc.reshape(len(Pr), -1)

    TOP = 1
    indices = []
    for i in range(len(mcc)):
        best_idx = mcc[i].argmax()
        coord = insitu_coords[insitu_coords.index ==
                              best_idx][['xcoord', 'ycoord', 'zcoord']].values
        dist = scipy.spatial.distance.cdist(
            coord, insitu_coords[['xcoord', 'ycoord', 'zcoord']].values)[0]
        top_idx = dist.argsort()[:TOP]
        top_idx = [e+1 for e in top_idx]
        indices += [top_idx]
    print(indices)
    print(len(indices))
    np.savetxt('largest_20.txt', indices)


genes = feature_selection(20)
find_largest_MCC(genes)
