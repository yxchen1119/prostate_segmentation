import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import data_processing
import model
#args

parser = argparse.ArgumentParser(description='Prostate Segmentation Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epoch', default=20, type=int, help='number of iterations')
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
net = model.Unet(in_dim=1, out_dim=1, num_filters=4)
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


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

#Train
def train(epoch, f):
    print('\nEpoch: %d' % epoch)
    net.train()
    running_loss = 0
    for batch_idx, (inputs, targets) in enumerate(Train_data_loader):
        inputs = inputs.float().unsqueeze(1)
        targets = targets.int().unsqueeze(1)
        inputs, targets = inputs.cuda(), targets.cuda()# get the inputs; data is a list of [inputs, labels]
        optimizer.zero_grad() # zero the parameter gradients
        outputs = net(inputs)
        weight = get_weight(targets)
        loss = F.cross_entropy(outputs, targets, weight=weight) #rb: the ratio of the numbers of voxels in the prostate and non-prostate regions in mini-batch b
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 10))

            f.write('[%d, %5d] loss: %.3f \n' %
                  (epoch + 1, batch_idx + 1, running_loss / 10))

            running_loss = 0.0

    print("==>Waiting Test")
    for batch_idx, (inputs, targets) in enumerate(Val_data_loader):
        inputs = inputs.float().unsqueeze(1)
        targets = targets.int().unsqueeze(1)
        #inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        dice = dice_coeff(outputs, targets)
        print("[%d, %5d] dice: %.3f" %
              (epoch + 1, batch_idx + 1, dice))
        
        f.write("[%d, %5d] dice: %.3f\n" %
                (epoch + 1, batch_idx + 1, dice))

    print('Saving model......')
    torch.save(net.state_dict(), './model/net_%03d.pth' % epoch + 1)


if __name__ == "__main__":
    with open('log.txt', 'w') as f:
        for epoch in range(start_epoch, start_epoch + 10):
            train(epoch, f)
            args.lr = args.lr * (1 - epoch / args.epoch) ** 0.9
        print("==>Training Finished, TotalEPOCH=%d" % epoch)