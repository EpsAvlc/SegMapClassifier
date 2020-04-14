# !/home/caoming/Projects/SegMapClassifier/venv/bin/python3.6
# -*- coding: utf-8 -*-

import numpy as np
from dataset import Dataset
from preprocessors import Preprocessor
from generator import Generator
from net import SegMapNet
import sys



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
    # import tensorwatch as tw

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    segmap_net = SegMapNet(data18.n_classes)
    segmap_net.initialize_weights()
    segmap_net.to(device)
    # segmap_net = segmap_net.float()
    # tw.draw_model(segmap_net, [1, 1, 32, 32, 16])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(segmap_net.parameters(), lr=0.0002)

    batches = np.array([1] * gen_train.n_batches)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir='/home/caoming/Projects/SegMapClassifier/logs', comment='segmap_torch')
    for epoch in range(256):
        running_loss = 0.0
        correct = 0
        np.random.shuffle(batches)

        for step, train in enumerate(batches):
            optimizer.zero_grad()

            batch_segments, batch_classes = gen_train.next()
            if batch_segments.shape[0] == 1:
                # only have one sample in a batch
                continue

            batch_segments = torch.from_numpy(batch_segments).to(device)
            batch_segments.requires_grad = True
            batch_classes = torch.from_numpy(batch_classes).to(device)

            scales_torch = torch.from_numpy(np.array(preprocessor.last_scales)).to(device)
            scales_torch.requires_grad = True
            output = segmap_net(batch_segments.float(), scales_torch.float())
            ## disp accuracy

            
            # print('correct: %d' % correct)
            ##
            loss = criterion(output, batch_classes)
            loss.backward()
            # print(segmap_net.fc2.weight)
            optimizer.step()
            # print(segmap_net.fc2.weight)
            running_loss += loss.item()
            predict_classes = torch.argmax(output, 1)
            for i in range(predict_classes.size()[0]):
                if predict_classes[i] == batch_classes[i]:
                    correct = correct + 1
            if step % 100 == 99:    # print every 2000 mini-batches
                acc = float(correct) / 100.0 / preprocessor.batch_size
                print('[%d, %5d] loss: %.3f, acc: %f, correct: %d' %
                    (epoch + 1, step + 1, running_loss / 100, 
                    acc*100, correct))
                writer.add_scalar('Train/Loss', running_loss, epoch)
                writer.add_scalar('Train/Acc', acc, epoch)
                running_loss = 0.0 
                correct = 0
        
        torch.save(segmap_net.state_dict(), '/home/caoming/Projects/SegMapClassifier/models/segmap_ ' 
        + str(epoch)+'.pkl')
    writer.close()
    print('Finished Training')


