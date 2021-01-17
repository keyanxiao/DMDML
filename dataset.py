# -*- coding: <encoding name> -*-
"""
ClassificationDataset
    -- 分类数据集
"""
from __future__ import print_function, division
from torch.utils.data import Dataset
import h5py
import numpy as np
from sklearn import preprocessing
from collections import Counter

################################################################################
# Triplet数据集
################################################################################
class TripletDataset(Dataset):
    def __init__(self, num_sub_dataset):
        """
        :param num_sub_dataset(int) 子集数量
        :param mode(str): 模式
        """
        # min - max标准化
        min_max_scaler = preprocessing.MinMaxScaler()
        # 样本种类数
        self.num_class = 10
        # 特征数
        self.feature = 8

        # 训练模式
        self.num_sub_dataset = num_sub_dataset # 子集数量
        x_list, y_list = [], [] # 训练数据集合
        # 加载训练数据
        with h5py.File('Data/yeast_classification_subset_train.h5', 'r') as hf:
            for i in range(self.num_sub_dataset):
                data = np.asarray(hf[str(i)])
                x, y = data[:, :-1], data[:, -1]
                # x = min_max_scaler.fit_transform(x)
                x_list.append(x)
                y_list.append(y)
        # 存储每个类别的数据
        data_list = []
        for i in range(self.num_sub_dataset):
            x, y = x_list[i], y_list[i]
            data = {}
            for i in range(len(y)):
                k = int(y[i])
                v = x[i]
                v = v[np.newaxis, :]
                if k in data.keys():
                    data[k] = np.concatenate((data[k], v), axis=0)
                else:
                    data[k] = v
            data_list.append(data)
        self.data_list = data_list
        self.num = len(self.data_list[0][self.num_class-1])

    def __getitem__(self, index):
        x_list, y_list = [], []
        for i in range(self.num_sub_dataset):
            data = self.data_list[i]
            x = np.zeros((0, self.feature))
            y = np.asarray([i for i in range(self.num_class)])
            nums = [len(data[i]) for i in range(self.num_class)]
            for i in range(self.num_class):
                index = np.random.choice(nums[i])
                x = np.concatenate((x, data[i][index: index + 1]), axis=0)
            x_list.append(x)
            y_list.append(y)
        return np.asarray(x_list), np.asarray(y_list)

    def __len__(self):
        return self.num


################################################################################
# Difference数据集
################################################################################`
class DifferenceDataset:
    def __init__(self):
        # # min - max标准化
        # min_max_scaler = preprocessing.MinMaxScaler()
        # 样本种类数
        self.num_class = 10
        # 特征数
        self.feature = 8
        # 文件路径
        path = 'Data/yeast_classification_train.h5'


        # 数据集合
        data_set = np.zeros((0, self.feature + 1))
        # 加载验证数据
        with h5py.File(path, 'r') as hf:
            for i in range(self.num_class):
                data = np.asarray(hf[str(i)])
                # 添加类别
                label = np.asarray([i for m in range(len(data))])
                label = label[np.newaxis, :]
                data = np.c_[data, label.T]
                data_set = np.concatenate((data_set, data), axis=0)
        x, y = data_set[:, :-1], data_set[:, -1]
        # x = min_max_scaler.fit_transform(x)

        # 存储每个类别的数据
        data = {}
        for i in range(len(y)):
            k = int(y[i])
            v = x[i]
            v = v[np.newaxis, :]
            if k in data.keys():
                data[k] = np.concatenate((data[k], v), axis=0)
            else:
                data[k] = v
        self.data = data
        self.nums = [len(self.data[i]) for i in range(self.num_class)]
        # # 基于滑动窗口采样，获取各类数据
        # self.index_dict = dict([(i, 0) for i in range(0, self.num_class)])

    def getsamples(self, batch_size):
        x = np.zeros((0, self.feature))
        y = []
        indexs = [i for i in range(self.num_class)]
        for k in range(batch_size):
            y.extend(indexs)
        y = np.asarray(y)
        for k in range(batch_size):
            for i in range(self.num_class):
                # index = self.index_dict[i]
                index = np.random.choice(self.nums[i])
                x = np.concatenate((x, self.data[i][index: index + 1]), axis=0)
                # self.index_dict[i] = (index + 1) % self.nums[i]

        return x, y


################################################################################
# KNN数据集
################################################################################
class KNNDataset(Dataset):
    def __init__(self, mode='train'):
        """
        :param mode(str): 模式
        """
        # 模式
        self.mode = mode
        # min - max标准化
        min_max_scaler = preprocessing.MinMaxScaler()
        # 样本种类数
        self.num_class = 10
        # 特征数
        self.feature = 8

        # 文件路径
        if mode == 'train':
            path = 'Data/yeast_classification_train.h5'
        else:
            path = 'Data/yeast_classification_val.h5'
        # 数据集合
        data_set = np.zeros((0, self.feature + 1))
        # 加载验证数据
        with h5py.File(path, 'r') as hf:
            for i in range(self.num_class):
                data = np.asarray(hf[str(i)])
                # 添加类别
                label = np.asarray([i for m in range(len(data))])
                label = label[np.newaxis, :]
                data = np.c_[data, label.T]
                data_set = np.concatenate((data_set, data), axis=0)
        x, y = data_set[:, :-1], data_set[:, -1]
        # x = min_max_scaler.fit_transform(x)
        self.x, self.y = x, y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def counter(self):
        return Counter(self.y)