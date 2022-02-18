#from dataset.dataset_soma import Soma_DataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
#from utilsa import logger, init_util, metrics,common
from dataset.predict_dataset import predict_dataset
import SimpleITK as sitk
import os
import numpy as np
from models.myunet import unet3d
import glob as glob
#from utilsa.common import load_file_name_list
#from collections import OrderedDict
#import matplotlib.pyplot as plt


def write_name_list(self, name_list, file_name,score):
    f = open(self.fixed_path + file_name, 'w')
    for i in range(len(name_list)):
        f.write(str(name_list[i]) +'    '+score +"\n")
    f.close()

def predict(model, dataset, save_path):
    predict_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    print(len(predict_loader))
    model.eval()
    f = open(save_path + '/score.txt', 'w')
    with torch.no_grad():
        for idx,(data) in tqdm(enumerate(predict_loader),total=len(predict_loader)):
            data= data.float()
            data= data.to(device)
            output = model(data)
            img = output.cpu().detach().numpy()
            score=np.array(list(img))
            score[score>0.99]=0
            score[score<0.01]=0
            score=np.where(score>0.5,1-score,score)

            #exist=(score!=0)
            #print(exist.sum())
            confidence_score=score.sum()/(128*128*128)
            print(confidence_score)
            img[img <= 0.5] = 0.
            img[img > 0.5] = 1.
            img = img * 255
            #print(img.shape)
            a = np.array(img, dtype='uint8')
            a = np.squeeze(a, axis=0)
            print(a.shape)
            #a = np.squeeze(a, axis=0)
            '''b=a[0]+a[1]
            b[b<=0]=0
            img = sitk.GetImageFromArray(b)'''
            img = sitk.GetImageFromArray(a[0])
            sitk.WriteImage(img, os.path.join(save_path, 'seg_' + dataset.filename_list[idx] ))
            f.write(dataset.filename_list[idx]+'        confidence score:'+str(confidence_score) + "\n")
        f.close()


if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')

        # model info
    #model = UNet(1, [32, 48, 64, 96, 128], 2, net_mode='3d',conv_block=RecombinationBlock).to(device)
    model = unet3d(1,[8,16,32,64,128],1).to(device)
    #model = unet3d(1, [8, 16, 32, 64, 128], 2).to(device) #2021.10.13
    #ckpt = torch.load('./output/{}/best_model.pth'.format(args.predict)) #2021.10.13 model:resizeadd result_path:'D:/A_predictcsz/bad_predict'
    ckpt = torch.load('./output/best_model.pth')
    model.load_state_dict(ckpt['net'])

    # data info
    predict_data_path = r'E:/soma_img_crop_uint8/soma_img_crop_uint8/*'
    #result_save_path = r'./output/{}/result'.format(args.save)
    #result_save_path=r'D:/A_predictcsz/augmor_predict'
    result_save_path=r'E:/soma_img_crop_uint8/seg_soma_img_crop_uint8/'

    '''if not os.path.exists(result_save_path): os.mkdir(result_save_path)
    datasets = predict_dataset(predict_data_path)'''
    predict_root_files=glob.glob(predict_data_path)
    for predict_root_file in predict_root_files:
        seg_root_filename=predict_root_file.split('\\')[-1]
        result_final_save_path=result_save_path+'/'+seg_root_filename+'/'
        #if not os.path.exists(result_save_path+'/'+seg_root_filename): os.mkdir(result_save_path+'/'+seg_root_filename)
        datasets=predict_dataset(predict_root_file+'/')
        #print(result_final_save_path)
        predict(model,datasets,result_final_save_path)


    #for dataset,file_idx in datasets:
    #predict(model, datasets,result_save_path)