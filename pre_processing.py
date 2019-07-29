import SimpleITK as sitk
from os import listdir
from os.path import  isfile,join

def N4_Bias_correction(input_img_path):
    input_img = sitk.ReadImage(input_img_path)
    mask_img = sitk.OtsuThreshold(input_img, 0, 1, 200)
    #灰度图像二值化阈值就是说灰度图的灰度是从0~255.设一个阈值，超过它的就直接当做255，没超过的就直接当做0，这样图像会被转化成黑白图，只有0和255两个值来表示
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    output_img = corrector.Execute(input_img, mask_img)
    output_img = sitk.Cast(output_img, sitk.sitkInt16)
    sitk.WriteImage(output_img, "./Corrected_Data/input_img_path")

if __name__ == '__main__':

    root = './TrainData'

    fileList = [f for f in listdir(root) if
                isfile(join(root, f)) and 'raw' not in f]
    print(fileList)
    for img_path in fileList:
        N4_Bias_correction(root + '/' + img_path)


