import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from utilsa.common import *
import matplotlib.pyplot as plt


def fix_data(raw_path,fixed_path):
    print('the raw dataset total numbers of samples is :', len(os.listdir(raw_path + 'data')))
    for data_file in os.listdir(raw_path + 'data/'):
        print(data_file)
        data = sitk.ReadImage(os.path.join(raw_path + 'data/', data_file), sitk.sitkInt8)
        data_array = sitk.GetArrayFromImage(data)
        data_array = np.array(data_array, dtype='uint8')
        #print(data_array.shape)


        #data_array[data_array<0.01]=0
        #data_array[data_array>0.99]=1
        #data_array = norm_img(data_array)

        arr = data_array.flatten()
        x=sorted(arr)
        x=np.array(x)
        z=x[2070000]
        #x_begin=np.where(x0)
        print(x)
        print(z)


        data_array[data_array<z]=0
        data_array[data_array>=z]=1

        data_array*=255
        data_array=np.array(data_array,dtype='uint8')



        new_data = sitk.GetImageFromArray(data_array)


        sitk.WriteImage(new_data, os.path.join(fixed_path + 'data/', data_file))

def main():
    raw_path='E:/soma_seg/correct/raw1/fixed_data/'
    fixed_path='E:/mean/'
    fix_data(raw_path,fixed_path)


if __name__ == '__main__':
    main()