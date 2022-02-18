import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from utilsa.common import *


class Soma_fix:
    def __init__(self, raw_dataset_path, fixed_dataset_path):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path

        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(self.fixed_path + 'data/')
            os.makedirs(self.fixed_path + 'label/')

        self.fix_data()  # 对原始图像进行修剪并保存
        self.write_train_val_test_name_list()  # 创建索引txt文件

    def fix_data(self):

        print('the raw dataset total numbers of samples is :', len(os.listdir(self.raw_root_path+ 'data')))
        for data_file in os.listdir(self.raw_root_path + 'data/'):
            print(data_file)
            data = sitk.ReadImage(os.path.join(self.raw_root_path + 'data/', data_file), sitk.sitkInt8)
            data_array = sitk.GetArrayFromImage(data)

            seg = sitk.ReadImage(os.path.join(self.raw_root_path + 'label/', 'seg_' + data_file),
                                 sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array = -seg_array * 255
            #data_array = norm_img(data_array)

            print(data_array.shape, seg_array.shape)



            new_data = sitk.GetImageFromArray(data_array)
            new_seg = sitk.GetImageFromArray(seg_array)

            sitk.WriteImage(new_data, os.path.join(self.fixed_path + 'data/', data_file))
            sitk.WriteImage(new_seg,
                            os.path.join(self.fixed_path + 'label/', 'seg_'+data_file))

    def write_train_val_test_name_list(self):
        data_name_list = os.listdir(self.fixed_path + "data")
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        train_rate = 0.9
        val_rate = 0.1

        assert val_rate + train_rate == 1.0
        train_name_list = data_name_list[0:int(data_num * train_rate)]
        val_name_list = data_name_list[int(data_num * train_rate):int(data_num * (train_rate + val_rate))]

        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.fixed_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()


def main():
    raw_dataset_path = 'D:/DLrawdata/batch1/'
    fixed_dataset_path = 'D:/DLtrain/fixed_data/'

    Soma_fix(raw_dataset_path, fixed_dataset_path)


if __name__ == '__main__':
    main()
