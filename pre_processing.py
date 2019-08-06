import SimpleITK as sitk
from os import listdir
from os.path import  isfile,join

def N4_Bias_correction(input_img_path):
    root = './Corrected_Data/'
    input_img = sitk.ReadImage('./TrainData/' + input_img_path)

    for i in range(5):
        mask_img = sitk.OtsuThreshold(input_img, 0, 1, 200)
    #灰度图像二值化阈值就是说灰度图的灰度是从0~255.设一个阈值，超过它的就直接当做255，没超过的就直接当做0，这样图像会被转化成黑白图，只有0和255两个值来表示
        input_img = sitk.Cast(input_img, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        output_img = corrector.Execute(input_img, mask_img)
        input_img = sitk.Cast(output_img, sitk.sitkInt16)

    sitk.WriteImage(input_img, root + input_img_path)
    print('processing done')
    return


if __name__ == '__main__':

    root = './TrainData'

    fileList = [f for f in listdir(root) if
                isfile(join(root, f)) and 'raw' not in f]
    print(fileList)
    for img_path in fileList[11:]:
        print('processing:' + img_path)

        N4_Bias_correction(img_path)


