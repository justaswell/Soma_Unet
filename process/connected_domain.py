import SimpleITK as sitk
import numpy as np
import glob as glob
import os



#def connected_domain(image, mask=True):
def connected_domain(image):
    label_image = sitk.ConnectedComponent(image)  # image必须是二值图像
    label_image_array = sitk.GetArrayFromImage(label_image)
    output = np.zeros_like(label_image_array)
    print(output.shape)
    x,y,z=output.shape
    #if label_image_array[int(x/2),int(y/2),int(z/2)]!=0:
    point = [int(z/2),int(y/2),int(x/2)]
    '''else:
        for i in range(20):
            if label_image_array[int(x/2)-10+i,int(y/2),int(z/2)]!=0:
                point = [int(z/2),int(y/2),int(x/2)-10+i]
                break'''
    #point = [132, 127,62]
    print(point)
    value = label_image.GetPixel(point)
    if value != 0:
        output[label_image_array == value] = 1
    #output = sitk.GetImageFromArray(output,sitk.sitkInt8)
    '''output.SetOrigin(image.GetOrigin())
    output.SetSpacing(image.GetSpacing())
    output.SetDirection(image.GetDirection())'''
    return output
    '''cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)

    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask

    if mask:
        output = (output > 0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    return output'''

def main():
    raw_path="E:/soma_img_crop_uint8/seg_soma_img_crop_uint8/*"
    fixed_path="E:/soma_img_crop_uint8/seg_postprocess/"
    file_path=glob.glob(raw_path+'*')
    for file in file_path:
        file_name=file.split("\\")[-1]
        #print(file_name)
        post_files=glob.glob(file+'/*')
        #print(post_files)
        for post_file in post_files:
            post_file_name=post_file.split('\\')[-1]
            #print(fixed_path+'/'+file_name+'/'+post_file_name)
        #if not os.path.exists(fixed_path + '/' + file_name): os.mkdir(fixed_path + '/' + file_name)
            label = sitk.ReadImage(post_file,sitk.sitkInt8)
    
            output=connected_domain(label)
    
            output=np.array(output,dtype='uint8')
            output*=255
            output=sitk.GetImageFromArray(output)
    
            sitk.WriteImage(output,fixed_path+'/'+file_name+'/'+post_file_name)

#if not os.path.exists(result_save_path+'/'+seg_root_filename): os.mkdir(result_save_path+'/'+seg_root_filename)

if __name__ == '__main__':
    main()