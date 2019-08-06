import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

def conv3d_block(in_num, out_num):
    return nn.Sequential(
        nn.Conv3d(in_num, out_num, kernel_size=3, stride=1, padding=1),#padding=1,output size = input size
        nn.BatchNorm3d(out_num),
        nn.ReLU()
        )

def max_pooling3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

# Upconvolution
def trans_conv3d_block(in_num, out_num):
    return nn.Sequential(
        nn.ConvTranspose3d(in_num, out_num, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_num),
        nn.ReLU()
    )
#each stage has 2 conv_block
def conv3d_2_block(in_num, out_num, up):
    if(not up):
        return nn.Sequential(
	        conv3d_block(in_num, out_num),
	        conv3d_block(out_num, out_num*2)
	    )
    else:
        return nn.Sequential(
            conv3d_block(in_num, out_num),
            conv3d_block(out_num, out_num)
        )

class Unet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(Unet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
#down sampling
        self.down1 = conv3d_2_block(self.in_dim, self.num_filters, False)
        self.pool1 = max_pooling3d()
        self.down2 = conv3d_2_block(self.num_filters*2, self.num_filters*2, False)
        self.pool2 = max_pooling3d()
        self.down3 = conv3d_2_block(self.num_filters*4, self.num_filters*4, False)
        self.pool3 = max_pooling3d()

        self.bridge = conv3d_2_block(self.num_filters*8, self.num_filters*8, False)
#up sampling
        self.trans1 = trans_conv3d_block(self.num_filters*16, self.num_filters*16)
        self.up1 = conv3d_2_block(self.num_filters*24,self.num_filters*8, True)
        self.trans2 = trans_conv3d_block(self.num_filters*8, self.num_filters*8)
        self.up2 = conv3d_2_block(self.num_filters*12, self.num_filters*4, True)
        self.trans3 = trans_conv3d_block(self.num_filters*4, self.num_filters*4)
        self.up3 = conv3d_2_block(self.num_filters*6, self.num_filters*2, True)
#output
        self.out = conv3d_block(self.num_filters*2, self.out_dim)

    def forward(self, x):
        down_1 = self.down1(x)#->(1,8,128,128,128)
        pool_1 = self.pool1(down_1)#->(1,8,64,64,64)

        down_2 = self.down2(pool_1)#->(1,16,64,64,64)
        pool_2 = self.pool2(down_2)#->(1,16,32,32,32)

        down_3 = self.down3(pool_2)#->(1,32,32,32,32)
        #print("out size: {}".format(down_3.size())) debug
        pool_3 = self.pool3(down_3)#->(1,32,16,16,16)

        bridge = self.bridge(pool_3)#->(1,64,16,16,16)

        trans_1 = self.trans1(bridge)#->(1,64,32,32,32)
        #print("out size: {}".format(trans_1.size())) debug
        concat_1 = torch.cat([trans_1, down_3], dim=1)#->(1,96,32,32,32)
        up_1 = self.up1(concat_1)#->(1,32,32,32,32)

        trans_2 = self.trans2(up_1)#->(1,32,64,64,64)
        concat_2 = torch.cat([trans_2, down_2], dim=1)#->(1,48,64,64,64)
        up_2 = self.up2(concat_2)#->(1,16,64,64,64)

        trans_3 = self.trans3(up_2)#->(1,16,128,128,128)
        concat_3 = torch.cat([trans_3, down_1], dim=1)#->(1,24,128,128,128)
        up_3 = self.up3(concat_3)#->(1,8,128,128,128)

        out = self.out(up_3)#->(1,3,128,128,128)

        return out

'''
if __name__ == "__main__":
    image_size = 128
    #x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x = torch.rand(1, 3, image_size, image_size, image_size)
    print("x size: {}".format(x.size()))
    #x = torch.autograd.Variable(x)
    model = Unet(in_dim=3, out_dim=3, num_filters=4)
    #out = model(x)
    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(model, (x, ))
    #print(out.size())
'''