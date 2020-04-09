#
# Created on Mon Apr 06 2020
#
# Copyright (c) 2020 HITSZ-NRSL
# All rights reserved
#
# Author: EpsAvlc
#

#!/usr/bin/python3  
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

class SegMapNet(nn.Module):

    def __init__(self):
        super(SegMapNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(2, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2))
        self.fc1 = nn.Linear(64 * 5 * 5 * 1, 512)
        self.bn1 = nn.BatchNorm3d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 64)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(x)) 
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))

if __name__ == "__main__":
    net = SegMapNet()

