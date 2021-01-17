# -*- coding: <encoding name> -*-
"""

"""
from __future__ import print_function, division
import torch.nn as nn
import torch
from torchsummary import summary


################################################################################
# TripletModule
################################################################################
class TripletModule(nn.Module):
    def __init__(self):
        super(TripletModule, self).__init__()
        self.layers = nn.Sequential(nn.Linear(8, 6),
                                     #nn.ReLU(True),
                                     #nn.Linear(6, 6)
                                     )
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.015)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.layers(input)
        return output


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 选择在cpu或cuda运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 可视化ICNet
    model = TripletModule().to(device)
    summary(model, (8,))
    print(model)

    pass