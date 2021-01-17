# -*- coding: <encoding name> -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import torch
import random


################################################################################
# EarlyStopping机制.
# 如果给定patience后，验证集损失还没有改善，则停止训练.
################################################################################
class EarlyStopping:
    """
    EarlyStopping机制.
    """

    def __init__(self, patience=7, verbose=False):
        """
        初始化函数
        param:
            patience(int) -- 在上一次验证集损失改善后等待多少epoch
            verbose(bool) -- 是否打印信息
        :param verbose:
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score <= self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """
        当验证集损失下降时，存储模型
            param:
             val_loss -- 验证集损失
        """
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        self.val_loss_min = val_loss


################################################################################
# 设置随机种子.
################################################################################
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


################################################################################
# 根据 epoch 改变学习率.
################################################################################
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    if (epoch + 1) % 150 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    pass
