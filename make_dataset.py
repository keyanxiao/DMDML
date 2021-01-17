# -*- coding: <encoding name> -*-
"""

"""
from __future__ import print_function, division
import os
import numpy as np
import h5py
from imblearn.over_sampling import SMOTE
import math

################################################################################
# 划分数据集，存储相同类别
################################################################################
def __divide_dataset():
    # 数据集存储文件
    path = 'Data/yeast.data'
    if not os.path.isfile(path):
        raise Exception('数据集存储文件不存在!')

    # 存储每个类别的数据
    data = {}

    ## 划分数据集
    with open(path, 'r') as reader:
        for line in reader.readlines():
            line = line.strip()
            line = line.split()
            k = line[-1]
            v = np.asarray([float(x) for x in line[1:-1]])
            v = v[np.newaxis, :]
            if k in data.keys():
                data[k] = np.concatenate((data[k], v), axis=0)
            else:
                data[k] = v

    # 按样本数量排序
    data = dict(sorted(data.items(), key=lambda item: len(item[1])))

    # 存储数据集
    with h5py.File('Data/yeast_classification.h5', 'w') as hf:
        cnt = 0
        for k, v in data.items():
            hf[str(cnt)] = v
            print(cnt, k, len(v))
            cnt += 1

    print("划分数据集已完成")


################################################################################
# 统计数据集相关信息，输出每个类别的个数、每个特征的最大值和最小值以及占比
# 0 MIT 244 || 1 NUC 429 || 2 CYT 463 || 3 ME1 44 || 4 EXC 35
# 5 ME2 51 || 6 ME3 163 || 7 VAC 30 || 8 POX 20 || 9 ERL 5
################################################################################
def __statistic_dataset():
    # 总样本数
    # amount = 1484
    amount = 1484
    # amount = 293
    # 特征数
    feature = 8
    # 样本种类数
    category = 10
    # 特征最小值、最大值
    val = [[float("inf"), 0.] for i in range(feature)]

    # 获取特征最小值、最大值
    with h5py.File('Data/yeast_classification_train.h5', 'r') as hf:
        for i in range(category):
            data = np.asarray(hf[str(i)])
            print(i, data.shape[0], "{:.4f}".format(data.shape[0] / amount))
            for i in range(feature):
                val[i][0] = min(val[i][0], np.min(data[:, i]))
                val[i][1] = max(val[i][1], np.max(data[:, i]))

    # 打印特征最小值、最大值
    for i in range(feature):
        print("[{}--{}]".format(val[i][0], val[i][1]), end=' ')
    print('')


################################################################################
# 将数据集按比例划分为训练集和验证集，五折交叉验证
################################################################################
def __divide_train_val_dataset(ratio = 0.2, fold = 0):
    """
    :param ratio(float)
    :param fold(int)
    :return:
    """
    # 样本种类数
    category = 10

    # 按比例划分训练集和验证集
    with h5py.File('Data/yeast_classification.h5', 'r') as reader:
        # 保存验证集
        with h5py.File('Data/yeast_classification_val.h5', 'w') as writer:
            for i in range(category):
                # 读取数据
                data = np.asarray(reader[str(i)])
                # 数据集大小
                len = data.shape[0]
                # 获取单折样本数量
                stride = int(len * ratio)
                # 保存验证集
                writer[str(i)] = data[stride * fold : stride * (fold + 1)]

        # 保存训练集
        with h5py.File('Data/yeast_classification_train.h5', 'w') as writer:
            for i in range(category):
                # 读取数据
                data = np.asarray(reader[str(i)])
                # 数据集大小
                len = data.shape[0]
                # 获取单折样本数量
                stride = int(len * ratio)
                # 保存训练集
                writer[str(i)] = np.concatenate((data[: stride * fold], data[stride * (fold + 1) :])
                                        ,axis=0)
    print("划分数据集已完成")


################################################################################
# 生成子数据集
################################################################################
def __generate_sub_dataset():
    # 数据
    data_set = []
    # 样本种类数
    num_classes = 10
    # 特征数
    feature = 8
    # 少类数据增强比例
    ratio = 0.5

    # 读取数据
    with h5py.File('Data/yeast_classification_train.h5', 'r') as hf:
        for i in range(num_classes):
            data = np.asarray(hf[str(i)])
            # 添加类别
            label = np.asarray([i for m in range(len(data))])
            label = label[np.newaxis, :]
            data = np.c_[data, label.T]
            data_set.append(data)

    # 获取占比最大类别的样本个数
    max_sample_amount = len(data_set[num_classes - 1])
    # 获取占比最小类别的样本个数
    # min_sample_amount = len(data_set[0])
    # 设置子集每个类别数量
    # threshold = (min_sample_amount + max_sample_amount) // 2
    if num_classes % 2 == 0:
        threshold = round(math.sqrt(len(data_set[num_classes // 2 - 1]) * len(data_set[num_classes // 2])))
    else:
        threshold = len(data_set[num_classes // 2])
    # 设置子集数量
    #threshold=len(data_set[9])-1
    #threshold = 58
    num = max_sample_amount // threshold
    if max_sample_amount % threshold != 0:
        num += 1
    print("threshold: {}, num: {}".format(threshold, num))

    # 增强少类数据
    less_data = np.zeros((0, feature + 1)) # 增强后的少类数据
    k_neighbors = 5 # SMOTE算法超参数
    end_index = 0 # 少类截止索引
    while len(data_set[end_index]) <= threshold:
        end_index += 1
    # 增强少类
    for i in range(0, end_index):
        amo = data_set[i].shape[0] # 少类样本数量
        if amo < threshold and amo > k_neighbors * 3: # 使用SMOTE算法增强数据，增强数量最多为原来的一半
            data = np.zeros((amo + min(int(amo * ratio), threshold - amo), feature))
            label = np.asarray([-1 for i in range(len(data))])
            label = label[np.newaxis, :]
            data = np.c_[data, label.T]
            data = np.concatenate((data, data_set[i]), axis=0)
            x, y = data[:, :-1], data[:, -1]
            x, y = SMOTE(random_state=1, k_neighbors=k_neighbors).fit_sample(x, y)
            y = y[np.newaxis, :]
            data = np.c_[x, y.T]
            # 少类增强后的结果
            less_data = np.concatenate((less_data, data[len(data) // 2 :]), axis=0)
        else:
            less_data = np.concatenate((less_data, data_set[i]), axis=0)

    # 多类索引启始下标
    start_index = dict([(i, 0) for i in range(end_index, num_classes)])
    # 基于滑动窗口对多类采样
    with h5py.File('Data/yeast_classification_subset_train.h5', 'w') as hf:
        for i in range(num):
            more_data = np.zeros((0, feature + 1))
            for k, v in start_index.items():
                data = data_set[k]
                end = (v + threshold) % len(data)
                if v < end:
                    more_data = np.concatenate((more_data, data[v:end, ]), axis=0)
                else:
                    more_data = np.concatenate((more_data, data[v:, ], data[:end, ]), axis=0)
                start_index[k] = end
            data = np.concatenate((less_data, more_data), axis=0)
            hf[str(i)] = data

    print("生成子数据集已完成")


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 划分数据集
    __divide_dataset()

    # 划分训练集和验证集
    __divide_train_val_dataset(ratio=0.2, fold=4)

    # 统计数据集相关信息
    __statistic_dataset()

    # 生成子数据集
    __generate_sub_dataset()

    pass
