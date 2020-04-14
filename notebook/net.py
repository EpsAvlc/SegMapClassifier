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

    def __init__(self, n_classes):
        super(SegMapNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.fc1 = nn.Linear(64 * 8 * 8 * 4, 512)
        # TODO: BN ERROR.
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, n_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        # initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))

        x = x.view(-1, 64 * 8 * 8 * 4)
        x = self.bn1(self.fc1(x))
        # x = self.relu(x) 
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        # x = self.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def initialize_weights(self):
        # print(self.modules())
 
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

if __name__ == "__main__":
    net = SegMapNet()

