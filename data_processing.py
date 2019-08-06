import SimpleITK as sitk
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
from os import listdir
from os.path import  isfile,join
import random
import transform

def load_itk(path,mode):
    itk_img = sitk.ReadImage(path)
    norm_img = Normalization(itk_img, mode) #resolution normalization
    img_array = RandomCrop(norm_img)

    #origin = np.array(itk_img.GetOrigin()) #读取图像原点坐标
    #spacing = np.array(itk_img.GetSpacing()) # 读取图像尺度信息

    return img_array

def Normalization(img, mode):
    base_res = (0.625, 0.625, 1.25)
    original_spacing = img.GetSpacing()
    retio = np.asarray(original_spacing) / base_res
    new_size = np.around(np.asarray(img.GetSize() * retio, dtype=float))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(base_res)
    resampler.SetSize(new_size.astype(dtype=int).tolist())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    norm_img = resampler.Execute(img)
    norm_img_array = sitk.GetArrayFromImage(norm_img)

    if(mode == 'image'):
        norm_img_array = norm_img_array.astype(dtype='float32')

    elif(mode == 'label'):
        norm_img_array = norm_img_array.astype(dtype='uint8')

    return norm_img_array



def RandomCrop(img):
    dim = img.shape

    x = random.randint(10, dim[0] - 32)
    y = random.randint(96, dim[1] - 96)
    z = random.randint(96, dim[2] - 96)
    cropImg = img[(x):(x + 32), (y):(y + 96), (z): (z + 96)]

    return cropImg


class Train_data(Dataset.Dataset):
    def __init__(self, root):  # 所有图片的绝对路径
        self.root = root
        self.transform = None

    def create_image_file_list(self):
        self.fileList = [f for f in listdir(self.root) if
                         isfile(join(self.root, f)) and 'segmentation' not in f and 'raw' not in f]

        print('FileList:' + str(self.fileList))

    def __getitem__(self, index):
        img_path = self.root + self.fileList[index]
        label_path = img_path[:-4] + '_segmentation.mhd'   #segmentation path

        image = load_itk(img_path, 'image')
        label = load_itk(label_path, 'label')

        self.transform = transform.get_transformer().create_transform()

        image = self.transform(image)
        label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.fileList)


def get_dataloader(root):
    dataSet = Train_data(root)
    dataSet.create_image_file_list()
    data_loader = DataLoader.DataLoader(dataSet, batch_size=16, shuffle=False, num_workers=0)
    return data_loader

'''
Traindata_loader = get_dataloader('./Corrected_Data/')

for batch_idx, (inputs, targets) in enumerate(Traindata_loader):

    print(inputs.size())


fileList = [f for f in listdir('./Corrected_Data/') if
                         isfile(join('./Corrected_Data/', f)) and 'segmentation' not in f and 'raw' not in f]
for f in fileList:
    a = load_itk('./Corrected_Data/'+f,'image')
    print(a)
'''
