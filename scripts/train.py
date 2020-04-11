#
# Created on Thu Apr 09 2020
#
# Copyright (c) 2020 HITSZ-NRSL
# All rights reserved
#
# Author: EpsAvlc
#

#!/usr/bin/python3  
# -*- coding: utf-8 -*-

import numpy as np
from dataset import Dataset
from preprocessors import Preprocessor
from generator import Generator
from net import SegMapNet


if __name__ == "__main__":
    preprocessor=Preprocessor(augment_angle=180,
        augment_remove_random_min=0.0,
        augment_remove_random_max=0.1,
        augment_remove_plane_min=0.0,
        augment_remove_plane_max=0.5,
        augment_jitter=0.0,
        align="eigen",
        voxelize=True,
        scale_method="fit",
        center_method="mean",
        scale=(8, 8, 4),
        voxels=(32, 32, 16),
        remove_mean=False,
        remove_std=False,
        batch_size=64,
        scaler_train_passes=1)
    
    data18 = Dataset("/home/caoming/Projects/SegMapClassifier/dataset/dataset18_big")
    # data18.load()

    segments, _, ids, n_ids, features, matches, labels_dict = data18.load(preprocessor=preprocessor)
    
    train_fold = np.ones(ids.shape, dtype=np.int)

    # print(len(np.unique(ids)))

    ## separating indexes on train and test parts
    for c in np.unique(ids):
        dup_classes = data18.duplicate_classes[ids == c]
        unique_dup_classes = np.unique(dup_classes)

        # choose for train the sequence with the largest last segment
        dup_sizes = []
        for dup_class in unique_dup_classes:
            dup_ids = np.where(dup_class == data18.duplicate_classes)[0]
            last_id = np.max(dup_ids)
            dup_sizes.append(segments[last_id].shape[0])

        dup_keep = np.argmax(dup_sizes)

        # randomly distribute the others
        for i, dup_class in enumerate(unique_dup_classes):
            if i != dup_keep:
                if np.random.random() < 0:
                    train_fold[duplicate_classes == dup_class] = 0

    train_ids = np.where(train_fold == 1)[0]
    preprocessor.init_segments(segments, ids,train_ids=ids)
    gen_train = Generator(
        preprocessor,
        train_ids,
        n_ids,
        train=True,
        batch_size=64,
        shuffle=True,
    )

    # Start training
    import torch
    import torch.optim as optim
    import torch.nn as nn

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    segmap_net = SegMapNet(data18.n_classes)
    segmap_net.to(device)
    segmap_net._initialize_weights()
    # segmap_net = segmap_net.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(segmap_net.parameters(), lr=0.001)

    batches = np.array([1] * gen_train.n_batches)
    for epoch in range(2):
        running_loss = 0.0
        np.random.shuffle(batches)

        for step, train in enumerate(batches):
            optimizer.zero_grad()

            batch_segments, batch_classes = gen_train.next()
            if batch_segments.shape[0] == 1:
                # only have one sample in a batch
                continue

            batch_segments = torch.from_numpy(batch_segments).to(device)
            batch_classes = torch.from_numpy(batch_classes).to(device)

            # print(batch_classes)
            # print(batch_segments.size())

            output = segmap_net(batch_segments.float())
            # print(output)
            loss = criterion(output, batch_classes)
            loss.backward()
            # print(segmap_net.fc2.weight)
            optimizer.step()
            # print(segmap_net.fc2.weight)
            running_loss += loss.item()
            # if step % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, step + 1, running_loss))
            running_loss = 0.0 
    print('Finished Training')


