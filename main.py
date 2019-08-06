import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import argparse
import transform
import data_processing
import model
#args

parser = argparse.ArgumentParser(description='Prostate Segmentation Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
Train_data_loader = data_processing.get_dataloader('./Train_data/')
Val_data_loader = data_processing.get_dataloader('./Val_data/')
print('==>done')

#Model
print('==> Building model..')
net = model.Unet(in_dim=3, out_dim=3, num_filters=4)
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-3, betas=(0.9, 0.999))
print('==>done')

def get_weight(target):
    target_array = target.data.cpu().numpy()
    fg_num = np.sum(target_array)
    bg_num = target_array.size - fg_num
    ratio = fg_num / bg_num
    weight = torch.ones(2)
    weight[0] = 1 / (1 - ratio)
    weight[1] = 1 /(ratio + 3e-6)
    weight = weight.cuda()

    return weight

#Train
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    running_loss = 0
    for batch_idx, (inputs, targets) in enumerate(Train_data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()# get the inputs; data is a list of [inputs, labels]
        optimizer.zero_grad() # zero the parameter gradients
        outputs = net(inputs)
        weight = get_weight(targets)
        loss = F.cross_entropy(outputs, targets, weight=weight) #rb: the ratio of the numbers of voxels in the prostate and non-prostate regions in mini-batch b
        loss.backward()
        optimizer.step()
        #args.lr
        running_loss += loss.item()
        if batch_idx % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

    print("==>Waiting Test")
    for batch_idx, (inputs, targets) in enumerate(Val_data_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)


    print("==>Training Finished, TotalEPOCH=%d" % epoch)

for epoch in range(start_epoch, start_epoch + 10):
    train(epoch)
