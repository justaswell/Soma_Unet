import numpy as np
import os
import torch
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from utilsa.common import *


class Soma_DataSet(Dataset):
    def __init__(self, dataset_path,mode=None):

        self.dataset_path = dataset_path

        if mode=='train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_name_list.txt'))
        elif mode =='val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')


    def __getitem__(self, index):
        data, target = self.get_train_batch_by_index(index=index)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self,index):
        img, label = self.get_np_data_3d(self.filename_list[index])
        return np.expand_dims(img,axis=0), label

    def get_np_data_3d(self, filename):
        data_np = sitk_read_raw(self.dataset_path +'data/'+ filename)
        #data_np = data_np-np.min(data_np)
        data_np=np.array(data_np,dtype='uint8')
        #print("rawimage's min and max:", np.min(data_np), np.max(data_np))
        #data_np=np.flip(data_np,axis=1)
        #print(data_np)
        data_np=norm_img(data_np)

        label_np = sitk_read_raw(self.dataset_path + 'label/seg_' + filename)
        label_np=np.array(label_np,dtype='uint8')

        #print("label's min and max:",np.min(label_np),np.max(label_np))
        label_np=norm_img(label_np)
        #label_np=-label_np
        label_np[label_np!=1]=0

        #print(label_np[0][64][64])
        #print(np.min(label_np[0]),' ',np.max(label_np[0]))
        '''p2 = filename.find('_raw')
        data_np = sitk_read_raw(self.dataset_path + 'image/' + filename, )
        data_np = norm_img(data_np)
        label_np = sitk_read_raw(
            self.dataset_path + 'GT/' + filename[0:p2] + '_task01_seg' + filename[p2 + 4:])
        label_np = norm_img(label_np)
        label_np[label_np==1]=1
        label_np[label_np<1]=0'''
        return data_np, label_np

# 测试代码
'''
import matplotlib.pyplot as plt
def main():
    fixd_path  = r'D:\DLresult'
    dataset = Soma_DataSet([128,128,128],1,fixd_path,mode='train')  #batch size
    data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
        target = to_one_hot_3d(target.long())
        print(data.shape, target.shape)
        plt.subplot(121)
        plt.imshow(data[0, 0, 0])
        plt.subplot(122)
        plt.imshow(target[0, 1, 0])
        plt.show()
if __name__ == '__main__':
    main()'''
