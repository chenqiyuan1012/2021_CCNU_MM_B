import scipy
from sklearn.neighbors import KNeighborsClassifier
import heapq
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

insitu_coords = pd.read_csv('data/geometry.txt', sep=' ')

y = insitu_coords[['xcoord', 'ycoord', 'zcoord']].values
bw = estimate_bandwidth(y, quantile=.01)
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(y)
# 使用Meanshift聚类算法，针对xyz数据进行聚类
cluster_centers = ms.cluster_centers_
# cluster_centers保存聚类中心
labels = ms.labels_
# labels保存的是对应的类别


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
genes = np.array(pd.concat([coef.sort_values().head(20)]))


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


# 将单元的聚类概率转换为聚类中心的加权坐标
def get_xyz(x):
    # 其中x是一个向量，每个维度对应属于每个类别的概率，例如x=[0.5 0.3 0.2]，那么表示属于第一类的概率是0.5……然后加权得到聚类中心的坐标
    ret = []
    for i in range(len(x)):
        v = np.average(cluster_centers, axis=0, weights=x[i, :])
        ret += [v]
    return np.array(ret)


insitu_bin = pd.read_csv('data/binarized_bdtnp.csv')
insitu_coords = pd.read_csv('data/geometry.txt', sep=' ')

cell_bin = pd.read_csv('data/dge_binarized_distMap.csv')
all_genes = list(cell_bin.index)
cell_bin = cell_bin.T
cell_bin.columns = all_genes

clf = KNeighborsClassifier()
# 使用带标签的数据来训练一个KNN分类模型
clf.fit(insitu_bin[genes].values, labels)
# pred是对应的需要预测坐标属于每个类别的概率
preds = clf.predict_proba(cell_bin[genes].values)
# 将pred从概率向量转化为坐标
preds = get_xyz(preds)
# 计算所求坐标到原来向量每个点的距离
preds_to_insitu_coords = scipy.spatial.distance.cdist(
    preds, insitu_coords[['xcoord', 'ycoord', 'zcoord']].values)

# 选择十个距离最短的
TOP = 10
indices = []
for i in range(len(preds_to_insitu_coords)):
    best_idx = preds_to_insitu_coords[i].argmin()
    coord = insitu_coords[insitu_coords.index ==
                          best_idx][['xcoord', 'ycoord', 'zcoord']].values
    dist = scipy.spatial.distance.cdist(
        coord, insitu_coords[['xcoord', 'ycoord', 'zcoord']].values)[0]
    top_idx = dist.argsort()[:TOP]
    top_idx = [e+1 for e in top_idx]
    indices += [top_idx]


dout = 'sc_20genes.csv'
with open(dout, 'w') as file_result:
    tmp = ['']
    for i in range(len(genes)):
        tmp += [genes[i]]
        if (i+1) % 10 == 0:
            file_result.write(','.join(tmp) + '\n')
            tmp = ['']
    for i in range(len(indices)):
        tmp = [i+1] + indices[i]
        file_result.write(','.join(map(str, tmp)) + '\n')
