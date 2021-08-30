import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def knn():
    largest = np.loadtxt('largest_20.txt')
    insitu_coords = pd.read_csv('data/geometry.txt', sep=' ')
    result = pd.read_csv('KNN_result/20.csv')

    print(result)
    print(insitu_coords)
    avg = []
    for i in range(1297):
        slice = np.array(result.iloc[i])  # 包含10个位置
        loc_ref = np.array(insitu_coords.iloc[int(largest[i])-1])  # MCC最小的位置
        dis = 0
        for j in range(10):
            loc_10 = np.array(insitu_coords.iloc[slice[j]-1])
            dis += np.linalg.norm(loc_10-loc_ref)
        avg.append(dis/10)
        print(len(avg))
    print(len(avg))
    return avg


def mcc():
    largest = np.loadtxt('largest_20.txt')
    insitu_coords = pd.read_csv('data/geometry.txt', sep=' ')
    result = pd.read_csv('MCC_result/20.csv')

    print(result)
    print(insitu_coords)
    avg = []
    for i in range(1297):
        slice = np.array(result.iloc[i])  # 包含10个位置
        loc_ref = np.array(insitu_coords.iloc[int(largest[i])-1])  # MCC最小的位置
        dis = 0
        for j in range(10):
            loc_10 = np.array(insitu_coords.iloc[slice[j]-1])
            dis += np.linalg.norm(loc_10-loc_ref)
        avg.append(dis/10)
        print(len(avg))
    print(len(avg))
    return avg


def xgb():
    largest = np.loadtxt('largest_20.txt')
    insitu_coords = pd.read_csv('data/geometry.txt', sep=' ')
    result = pd.read_csv('XGB_result/20.csv')

    print(result)
    print(insitu_coords)
    avg = []
    for i in range(1297):
        slice = np.array(result.iloc[i])  # 包含10个位置
        loc_ref = np.array(insitu_coords.iloc[int(largest[i])-1])  # MCC最小的位置
        dis = 0
        for j in range(10):
            loc_10 = np.array(insitu_coords.iloc[slice[j]-1])
            dis += np.linalg.norm(loc_10-loc_ref)
        avg.append(dis/10)
        print(len(avg))
    print(len(avg))
    return avg


avg_xgb = xgb()
avg_knn = knn()
avg_mcc = mcc()

print('ok')
plt.boxplot((avg_knn, avg_mcc, avg_xgb), labels=('KNN', 'MCC', 'XGBoost'))
plt.ylabel("Mean Euclidean distance")
plt.title("20 Genes")
plt.show()
print('okok')
