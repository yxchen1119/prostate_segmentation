import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import os

def load_itk(path):
    itk_img = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(itk_img)
    origin = np.array(itk_img.GetOrigin()) #读取图像原点坐标
    spacing = np.array(itk_img.GetSpacing()) # 读取图像尺度信息

    return img_array, origin, spacing

class Train_data(Dataset.Dataset):
    def __init__(self, root):  # 所有图片的绝对路径
        imgs = os.listdir(root)
        imgs.sort(key = lambda x:int(x[4:6]))
        #print(imgs)
        self.imgs = list()
        self.segs = list()
        for k in imgs:
            if k.split('.')[-1] == 'mhd':
                if not k.split('.')[0].split('_')[-1] == 'segmentation':
                    self.imgs.append([os.path.join(root, k)])
                else:
                    self.segs.append([os.path.join(root, k)])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_path = self.segs[index]
        itk_imag = load_itk(img_path)
        itk_label = load_itk(label_path)
        data = torch.from_numpy(itk_imag[0])
        label = torch.from_numpy(itk_label[0])

        #use gpu
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        return data, label

    def __len__(self):
        return len(self.imgs)


def get_train_dataloader(root):
    Train_dataSet = Train_data(root)  #'./TrainData'
    data_loader = DataLoader.DataLoader(Train_dataSet, batch_size=2, shuffle=False, num_workers=0)
    return data_loader