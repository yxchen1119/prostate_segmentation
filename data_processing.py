import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from os import listdir
from os.path import  isfile,join

def load_itk(path):
    itk_img = sitk.ReadImage(path)
    img_array = torch.from_numpy(sitk.GetArrayFromImage(itk_img)) #toTensor
    origin = np.array(itk_img.GetOrigin()) #读取图像原点坐标
    spacing = np.array(itk_img.GetSpacing()) # 读取图像尺度信息

    return img_array, origin, spacing

def pre_processing(image_path):
    itk_imag = load_itk(image_path)

    #########BIAS4
    #########Transform
    return data


class Train_data(Dataset.Dataset):
    def __init__(self, root):  # 所有图片的绝对路径
        self.root = root

    def create_image_file_list(self):
        self.fileList = [f for f in listdir(self.root) if
                         isfile(join(self.root, f)) and 'segmentation' not in f and 'raw' not in f]

        print('FileList:' + str(self.fileList))

    def __getitem__(self, index):
        img_path = self.fileList[index]
        label_path = img_path[:-4] + '_segmentation.mhd'   #segmentation path

        data = pre_processing(img_path)
        label = pre_processing(label_path)

        #use gpu
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        return data, label

    def __len__(self):
        return len(self.imgs)


def get_train_dataloader(root):
    Train_dataSet = Train_data(root)  #'./TrainData'
    Train_dataSet.create_image_file_list()
    data_loader = DataLoader.DataLoader(Train_dataSet, batch_size=2, shuffle=False, num_workers=0)
    return data_loader


daraLoader = get_train_dataloader('./TrainData')