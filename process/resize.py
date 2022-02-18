import pandas as pd
import numpy as np
import glob as glob
import SimpleITK as sitk
import os
from scipy import ndimage



def crop3D(raw_root_path,fixed_path,m):

    print('the raw dataset total numbers of samples is :', len(os.listdir(raw_root_path + 'data')))
    for data_file in os.listdir(raw_root_path + 'data/'):
        print(data_file)
        data = sitk.ReadImage(os.path.join(raw_root_path + 'data/', data_file), sitk.sitkInt8)
        ct_array = sitk.GetArrayFromImage(data)

        seg = sitk.ReadImage(os.path.join(raw_root_path + 'label/', 'seg_' + data_file),sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        #print(ct_array.shape,seg_array.shape)
        ct_array=ndimage.zoom(ct_array,m,order=0)
        seg_array=ndimage.zoom(seg_array,m,order=0)

        print(ct_array.shape,seg_array.shape)

        '''x = np.any(seg_array, axis=(0,1))
        #print(x.shape)
        x_start, x_end = np.where(x)[0][[0, -1]]
        y = np.any(seg_array, axis=(0,2))
        y_start,y_end= np.where(y)[0][[0, -1]]
        z = np.any(seg_array, axis=(1,2))
        z_start, z_end = np.where(z)[0][[0, -1]]

        print('z', z_start, z_end)
        print('y', y_start, y_end)
        print('x', x_start, x_end)


        x1=(x_start+x_end)//2
        y1 = (y_start + y_end) // 2
        z1 = (z_start + z_end) // 2

        #print(x1,y1,z1)
        ct_array=ct_array[z1-64:z1+64,y1-64:y1+64,x1-64:x1+64]
        seg_array=seg_array[z1-64:z1+64,y1-64:y1+64,x1-64:x1+64]'''
        z,y,x=ct_array.shape
        z=z//2
        y=y//2
        x=x//2
        if seg_array[z,y,x]:
            ct_array=ct_array[z-64:z+64,y-64:y+64,x-64:x+64]
            seg_array=seg_array[z-64:z+64,y-64:y+64,x-64:x+64]
            seg_array=np.flip(seg_array,axis=1)
        else:
            x0 = np.any(seg_array, axis=(0, 1))
            x_start, x_end = np.where(x0)[0][[0, -1]]
            y0 = np.any(seg_array, axis=(0, 2))
            y_start, y_end = np.where(y0)[0][[0, -1]]
            z0 = np.any(seg_array, axis=(1, 2))
            z_start, z_end = np.where(z0)[0][[0, -1]]

            print('z', z_start, z_end)
            print('y', y_start, y_end)
            print('x', x_start, x_end)

            x1 = (x_start + x_end) // 2
            y1 = (y_start + y_end) // 2
            z1 = (z_start + z_end) // 2

            # print(x1,y1,z1)
            if z1 - 64>=0 and z1 + 64<=128 and y1 - 64>=0 and y1 + 64<=128 and x1 - 64>=0 and x1 + 64<=128:
                ct_array = ct_array[z1 - 64:z1 + 64, y1 - 64:y1 + 64, x1 - 64:x1 + 64]
                seg_array = seg_array[z1 - 64:z1 + 64, y1 - 64:y1 + 64, x1 - 64:x1 + 64]
                seg_array = np.flip(seg_array, axis=1)
            else:
                ct_array = ct_array[z - 64:z + 64, y - 64:y + 64, x - 64:x + 64]
                seg_array = seg_array[z - 64:z + 64, y - 64:y + 64, x - 64:x + 64]
                seg_array = np.flip(seg_array, axis=1)
        #print(ct_array.shape,seg_array.shape)

        new_ct = sitk.GetImageFromArray(ct_array)
        new_seg = sitk.GetImageFromArray(seg_array)


        sitk.WriteImage(new_ct, os.path.join(fixed_path + 'data/', data_file))
        sitk.WriteImage(new_seg,os.path.join(fixed_path + 'label/', 'seg_' + data_file))

def crop3D_predict(raw_root_path,fixed_path,m):
    print('the raw dataset total numbers of samples is :', len(os.listdir(raw_root_path )))
    for data_file in os.listdir(raw_root_path ):
        print(data_file)
        data = sitk.ReadImage(os.path.join(raw_root_path, data_file), sitk.sitkInt8)
        ct_array = sitk.GetArrayFromImage(data)

        #print(ct_array.shape,seg_array.shape)
        ct_array=ndimage.zoom(ct_array,m,order=0)

        print(ct_array.shape)

        '''x = np.any(seg_array, axis=(0,1))
        #print(x.shape)
        x_start, x_end = np.where(x)[0][[0, -1]]
        y = np.any(seg_array, axis=(0,2))
        y_start,y_end= np.where(y)[0][[0, -1]]
        z = np.any(seg_array, axis=(1,2))
        z_start, z_end = np.where(z)[0][[0, -1]]

        print('z', z_start, z_end)
        print('y', y_start, y_end)
        print('x', x_start, x_end)


        x1=(x_start+x_end)//2
        y1 = (y_start + y_end) // 2
        z1 = (z_start + z_end) // 2

        #print(x1,y1,z1)
        ct_array=ct_array[z1-64:z1+64,y1-64:y1+64,x1-64:x1+64]
        seg_array=seg_array[z1-64:z1+64,y1-64:y1+64,x1-64:x1+64]'''
        z,y,x=ct_array.shape
        z=z//2
        y=y//2
        x=x//2

        ct_array=ct_array[z-64:z+64,y-64:y+64,x-64:x+64]

        new_ct = sitk.GetImageFromArray(ct_array)

        sitk.WriteImage(new_ct, os.path.join(fixed_path + 'data/', data_file))


def crop3D_result(data_path,fixed_path):                 #2021.10.19
    file = "E:/resize/AllbrainResolutionInfo.csv"
    data_name = data_path.split("\\")[-1]
    #print(data_name)
    brainID=data_name.split("_")[1]
    #print(brainID)
    brainID=int(brainID)
    data = pd.read_csv(file)
    data = np.array(data)
    #print(data)
    newdata = []
    x, _ = data.shape
    for i in range(x):
        if data[i][0] == brainID:
            newdata.append(data[i][1:])
            #print("find!")
    newdata = np.array(newdata)
    print(newdata)
    newdata = newdata.squeeze(axis=0)
    # print(newdata)
    newdata = newdata / 0.2
    m = [newdata[2], newdata[1], newdata[0]]
    print(data_path)

    data = sitk.ReadImage(data_path, sitk.sitkInt8)
    ct_array = sitk.GetArrayFromImage(data)

    #print(ct_array.shape,seg_array.shape)
    ct_array=ndimage.zoom(ct_array,m,order=0)

    print(ct_array.shape)

    z,y,x=ct_array.shape
    z=z//2
    y=y//2
    x=x//2

    ct_array=ct_array[z-64:z+64,y-64:y+64,x-64:x+64]

    new_ct = sitk.GetImageFromArray(ct_array)

    sitk.WriteImage(new_ct, os.path.join(fixed_path , data_name))




def script(_brainID):
    brainID=_brainID
    file="E:/resize/AllbrainResolutionInfo.csv"
    #raw_root_path='E:/resize/'+str(brainID)+'/'
    #fixed_path='D:/A_DLcsz/resize/'+str(brainID)+'/'
    raw_root_path = 'D:/A_predictcsz/bad_resize/' + str(brainID) + '/' #2021.10.13
    fixed_path='D:/A_predictcsz/bad_resize/' #2021.10.13
    data=pd.read_csv(file)
    data=np.array(data)
    newdata=[]
    x,_=data.shape
    for i in range(x):
        if data[i][0]==brainID:
            newdata.append(data[i][1:])
    newdata=np.array(newdata)
    newdata=newdata.squeeze(axis=0)
    #print(newdata)
    newdata=newdata/0.2
    m=[newdata[2],newdata[1],newdata[0]]
    print(m)
    crop3D_predict(raw_root_path,fixed_path,m)

if __name__ == '__main__':
    files=glob.glob('D:/A_predictcsz/bad_predict/*')
    fixed_path="D:/A_predictcsz/bad_predict_resize/"
    for file in files:
        crop3D_result(file,fixed_path)
    '''for file in files:
        id=file.split('\\')[1]
        if id=='data':
            continue
        print(id)
        script(int(id))'''