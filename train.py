# -*- coding: <encoding name> -*-
"""
train model
"""
from __future__ import print_function, division
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import TripletDataset, KNNDataset, DifferenceDataset
import torch
from model import TripletModule
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import label_binarize
from utils import setup_seed, EarlyStopping, adjust_learning_rate
from loss import batch_hard_triplet_loss
from sklearn.neighbors import KNeighborsClassifier
import torch.nn as nn

################################################################################
# 相关配置
################################################################################
# Set random seed for reproducibility
manualSeed = 1
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
setup_seed(manualSeed)
# 选择在cpu或cuda运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建存储模型的目录
if not os.path.isdir('Checkpoint'):
    os.mkdir('Checkpoint')
    os.mkdir('Checkpoint/models')


################################################################################
# train TripletModule model
################################################################################
def train_triplet_module():
    """
    :return:
    """
    # 设置超参数
    LR = 1e-5  # 学习率
    EPOCH = 15  # 训练轮数
    BATCH_SIZE = 10  # Class Batch size
    N_CLASS = 10  # 类别个数
    num_sub_dataset = 10  # 子集数量
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    resume = False  # 是否断点训练
    workers = 0  # Number of workers for dataloader
    margin = 1e-1  # triplet loss 超参数 margin
    k = 1
    # k in topk
    interval = 5 # diff loss和triplet loss间隔epoch
    balance = 4e-2 # diff loss和triplet loss间权重

    # 加载数据
    train_dataset = TripletDataset(num_sub_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    diff_dataset = DifferenceDataset()

    # 定义模型
    model_set = []
    for i in range(num_sub_dataset):
        model = TripletModule().float().to(device)
        model_set.append(model)

    # 定义优化器
    params_set = []
    for model in model_set:
        params = [{'params': model.parameters(), 'lr': LR}]
        params_set.extend(params)
    optimizer = optim.Adam(params_set, lr=LR)

    # 断点训练，加载模型权重
    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('Checkpoint'), 'Error: no Checkpoint directory found!'
        state = torch.load('Checkpoint/models/ckpt.pth')
        for i in range(num_sub_dataset):
            model = model_set[i]
            model.load_state_dict(state['net'][i])
        optimizer.load_state_dict(state['optim'])
        start_epoch = state['epoch']

    # 损失函数
    cosloss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    # 训练模型
    for epoch in range(start_epoch, EPOCH):
        print("####################################################################################")
        # 学习率调度机制
        adjust_learning_rate(optimizer, epoch)
        print('Learning rate is {}'.format(optimizer.param_groups[0]['lr']))
        ############################
        # 训练
        ############################
        # 训练模式
        for i in range(num_sub_dataset):
            model = model_set[i]
            model.train()
        # 迭代次数
        cnt = 0
        # triplet损失
        sum_triplet_loss = 0.
        # diff损失
        sum_diff_loss = 0.
        # 损失
        sum_loss = 0.
        for data in train_loader:
            cnt += 1

            # 加载Triplet数据集数据
            x, y = data
            batch_size = x.size(0)
            inputs, labels = torch.cat(tuple([x[:, i] for i in range(num_sub_dataset)]), dim=0),\
                                torch.cat(tuple([y[:, i] for i in range(num_sub_dataset)]), dim=0)
            inputs, labels = inputs.view((-1, inputs.size(-1))), labels.view(-1)
            inputs, labels = inputs.float().to(device), labels.int().to(device)

            # 梯度置零
            optimizer.zero_grad()

            # 前向传播、后向传播
            num_subset_sample = batch_size*N_CLASS # 每个子集对应batch的样本数量
            embeddings = torch.cat(tuple([model_set[i](inputs[num_subset_sample*i : num_subset_sample*(i+1)]) for i in range(num_sub_dataset)]),
                                   dim=0)

            triplet_loss = batch_hard_triplet_loss(k, num_subset_sample, labels, embeddings, margin=margin, device=device)
            # triplet loss
            sum_triplet_loss += triplet_loss.item()

            # 加载Difference数据集数据
            x, y = diff_dataset.getsamples(batch_size)
            inputs, labels = torch.from_numpy(x), torch.from_numpy(y)
            inputs, labels = inputs.float().to(device), labels.int().to(device)
            outputs = []
            outputs_sum = None
            for model in model_set:
                output = model(inputs)
                outputs.append(output)
                if outputs_sum is None:
                    outputs_sum = output
                else:
                    outputs_sum += output
            diff_loss = 0.
            for output in outputs:
                # diff_loss += torch.sum(torch.abs(cosloss(output, (outputs_sum-output)/(num_sub_dataset-1)))) / inputs.size(0)
                diff_loss += torch.sum(cosloss(output, (outputs_sum - output) / (num_sub_dataset - 1))) / inputs.size(0)
            diff_loss /= num_sub_dataset
            sum_diff_loss += diff_loss.item()

            loss = triplet_loss + balance * diff_loss
            sum_loss += loss.item()

            if (epoch + 1) % interval == 0:
                loss.backward()
            else:
                triplet_loss.backward()
            # 梯度更新
            optimizer.step()

            # 打印日志
            if cnt % 5 == 0 or cnt == len(train_loader):
                print('[%d/%d]--[%d/%d]\tTriplet Loss: %.6f\tDiff Loss: %.6f\tLoss: %.6f'
                      % (epoch + 1, EPOCH, cnt, len(train_loader), sum_triplet_loss / cnt, sum_diff_loss / cnt,
                         sum_loss / cnt))

        # 模型状态
        net_state_set = [model.state_dict() for model in model_set]
        # 保存断点模型
        state = {
            'net': net_state_set,
            'optim': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, './Checkpoint/models/ckpt.pth')


################################################################################
# train knn module
################################################################################
def train_knn_module():
    """
    :return:
    """
    # 设置超参数
    NUM_MODEL = 10 # 模型数量
    BATCH_SIZE = 150  # Batch size
    N_CLASS = 10  # 类别个数
    workers = 0  # Number of workers for dataloader
    n_neighbors = 6 # knn
    out_dim = 6  # 单个模型输出维度

    # 加载数据
    train_dataset = KNNDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=workers)
    val_dataset = KNNDataset(mode='test')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=workers)

    # 加载模型
    model_set = []
    # 加载权重
    state = torch.load('Checkpoint/models/ckpt.pth')
    for i in range(NUM_MODEL):
        model = TripletModule().float().to(device)
        # 加载权重
        model.load_state_dict(state['net'][i])
        model_set.append(model)

    # 获取训练数据embedding、labels
    x_trains, y_train = [np.zeros((0, out_dim)) for i in range(NUM_MODEL)], np.zeros((0, ))
    # 获取测试数据embedding、labels
    x_tests, y_test = [np.zeros((0, out_dim)) for i in range(NUM_MODEL)], np.zeros((0,))
    for model in model_set: # 测试模式
        model.eval()
    with torch.no_grad():
        # 训练数据
        for data in train_loader:
            # 加载数据
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.int().to(device)
            labels = labels.cpu().detach().numpy()
            y_train = np.append(y_train, labels)

            # 前向传播
            for i in range(NUM_MODEL):
                model = model_set[i]
                embeddings = model(inputs)
                embeddings = embeddings.cpu().detach().numpy()
                x_trains[i] = np.concatenate((x_trains[i], embeddings), axis=0)

        # 测试数据
        for data in val_loader:
            # 加载数据
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.int().to(device)
            labels = labels.cpu().detach().numpy()
            y_test = np.append(y_test, labels)

            # 前向传播
            for i in range(NUM_MODEL):
                model = model_set[i]
                embeddings = model(inputs)
                embeddings = embeddings.cpu().detach().numpy()
                x_tests[i] = np.concatenate((x_tests[i], embeddings), axis=0)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # 训练KNN
    model_set = []
    for i in range(NUM_MODEL):
        knn = KNeighborsClassifier(n_neighbors = n_neighbors)
        knn.fit(x_trains[i], y_train)
        model_set.append(knn)

    # 获取预测结果, 采用软投票策略
    y_pr_p = np.zeros((len(y_test), N_CLASS)) # 预测概率
    for i in range(NUM_MODEL):
        knn = model_set[i]
        y_pr_p += knn.predict_proba(x_tests[i])
    y_pr = np.argmax(y_pr_p, axis=1)  # 预测结果

    # 计算acc
    acc = accuracy_score(y_test, y_pr)
    # 计算f1
    f1 = f1_score(y_test, y_pr, average='macro')
    # 计算G-mean
    g_mean = geometric_mean_score(y_test, y_pr, average='macro')
    # 计算auc
    auc = roc_auc_score(label_binarize(y_test, np.arange(N_CLASS)), y_pr_p, average='macro')

    # 打印acc、f1、G-mean、auc
    print('ACC: %.4f\t F1: %.4f\n'
          'G-mean: %.4f\t AUC: %.4f' % (acc, f1, g_mean, auc))

    return acc, f1, g_mean, auc

################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    start_time = time.time()

    # train TripletModule model
    train_triplet_module()

    # train KnnModule model
    print('\n')
    print("################################################################################")
    train_knn_module()

    print("total time: %.2f minutes" % ((time.time() - start_time) / 60))
    pass
