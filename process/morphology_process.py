import numpy as np
import glob as glob
import SimpleITK as sitk
import os
from scipy import ndimage

def findedge(data_path,fixed_path):
    files=glob.glob(data_path+'*')
    for file in files:
        name=file.split('\\')[1]
        #print(name)
        label=sitk.ReadImage(file, sitk.sitkInt8)
        label_array=sitk.GetArrayFromImage(label)

        kernel1=np.ones((3,5,5),np.int8)
        kernel2=np.ones((1,3,3),np.int8)

        dilate=np.zeros((128,128,128))
        erose=np.zeros((128,128,128))

        ndimage.binary_dilation(label_array,kernel1,iterations=1,output=dilate)
        ndimage.binary_erosion(label_array,kernel2,iterations=1,output=erose)

        dilate=np.array(dilate,dtype='uint8')
        erose=np.array(erose,dtype='uint8')

        dilate[dilate!=0]=255
        erose[erose!=0]=255
        new=dilate-erose
        new[new<0]=255
        new_larry=label_array-new
        new=[new_larry,new]
        new=np.array(new,dtype='uint8')

        print(new.shape)

        #new_dilate=sitk.GetImageFromArray(dilate)
        #new_erose=sitk.GetImageFromArray(erose)
        dnew=sitk.GetImageFromArray(new)
        #sitk.WriteImage(new_dilate,fixed_path+'dilate.tiff')de
        #sitk.WriteImage(new_erose,fixed_path+'erose.tiff')
        sitk.WriteImage(dnew,fixed_path+name)

if __name__ == '__main__':
    data_path='D:/A_DLcsz/DLtrainf/fixed_data/labelraw/'
    fixed_path='D:/A_DLcsz/DLtrainf/fixed_data/label0/'
    '''files=glob.glob(data_path+'*')
    for file in files:
        name=file.split('\\')[1]
        if name=='elde':
            continue
        data_resize_path=data_path+name+'/label/'
        findedge(data_resize_path,fixed_path)'''
    findedge(data_path,fixed_path)