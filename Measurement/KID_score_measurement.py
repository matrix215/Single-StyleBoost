import numpy as np 
import os 
from os import path
import shutil, subprocess
from PIL import Image
from PIL import ImageFile

def KID_score(model, style):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    origin_pth_dir = f'./output_img/{style}'
    trg_pth_dir = f'./output_img/{model}/{style}'
    copy_pth = f'./output_img/{model}/{style}/all_images'
    origin_copy_pth = f'./output_img/{style}/all_images'
    npy_file = 'npy_file'
    
    #.npy 삭제
    if os.path.isfile(f'{trg_pth_dir}/{npy_file}.npy'):
        os.unlink(f'{trg_pth_dir}/{npy_file}.npy')
    if os.path.isfile(f'{origin_copy_pth}/{npy_file}.npy'):
        os.unlink(f'{origin_copy_pth}/{npy_file}.npy')
    
    #all_image dir 제거
    # if os.path.isdir(copy_pth):
    #     shutil.rmtree(copy_pth)
    # if os.path.isdir(origin_copy_pth):
    #     shutil.rmtree(origin_copy_pth)
        
    all_index_lst = os.listdir(trg_pth_dir)
    index_lst = [file for file in all_index_lst if not file.endswith(".npy")]
    
    #output_img/model/style/all_images 에 이미지 옮기기
    for index in index_lst:
        file_pth_dir = trg_pth_dir + '/' + index
        #각 prompt index 별로 계산하고 싶을 때, 해당 코드 아래는 주석처리하면됨. 대신 txt 쓰는 거에서 index 추가 해야 됨
        # proc = subprocess.Popen([f"python ./PyTorch-FID-score/fid_score.py {origin_pth_dir} {file_pth_dir} --model {model} --style {style}"],  shell=True)
        # out, _ = proc.communicate() 
        # =============================
        file_lst = os.listdir(file_pth_dir)

        if not path.isdir(copy_pth):
            os.makedirs(copy_pth)
        
        for file in file_lst:
            if not path.exists(copy_pth + '/' + file):
                shutil.copy(file_pth_dir+'/'+file, copy_pth+'/'+file)
    
    all_index_lst = os.listdir(origin_pth_dir)        
    index_lst = [file for file in all_index_lst if not file.endswith(".npy")]
    
    print('all target images: ',len(index_lst))
    
    index = 0
    except_index = 0
    test_index = 0
    for file in index_lst:
        if not path.isdir(origin_copy_pth):
            os.makedirs(origin_copy_pth)
        test_index += 1
        if not path.exists(origin_copy_pth + '/' + file):
            try:
                im = Image.open(origin_pth_dir +'/' + file)
                im = im.resize((512,512))
                im = im.convert('RGB')
                im.save(origin_copy_pth +'/'+ file) #str(index)+".png"
                index = index + 1
            except:
                except_index = except_index + 1 
    print('index: ',index)
    print('except_index: ',except_index)
    print('test index: ',test_index)
    
    all_imgs = os.listdir(copy_pth)
    imgs_len = len(all_imgs)
    all_imgs = np.array(all_imgs)
    img_list_np = []
    
    for i in all_imgs:
        img = Image.open(copy_pth + '/' + i)
        img_array = np.array(img)
        img_list_np.append(img_array)
    img_np = np.array(img_list_np).astype("float64")
    img_np = img_np.transpose(0,3,1,2)
    print('generated shape: ',np.shape(img_np))
    np.save(f'{trg_pth_dir}/{npy_file}',img_np)
    
    in_imgs_lst = os.listdir(origin_copy_pth)
    in_imgs = [file for file in in_imgs_lst if not file.endswith(".npy")]
    in_imgs = np.array(in_imgs)
    img_list_np = []
    
    for i in in_imgs:
        img = Image.open(origin_copy_pth + '/' + i)
        img_array = np.array(img)
        img_list_np.append(img_array)
    img_np = np.array(img_list_np).astype("float64")
    img_np = img_np.transpose(0,3,1,2)
    print('target shape: ',np.shape(img_np))
    np.save(f'{origin_copy_pth}/{npy_file}',img_np)
    
    proc = subprocess.Popen([f"python ./gan-metrics-pytorch/kid_score.py --true {origin_copy_pth}/{npy_file}.npy --fake {trg_pth_dir}/{npy_file}.npy --model2 {model} -c 1 --style {style} --length {imgs_len} --batch-size 20 --dims 768"],  shell=True)
    _ , _ = proc.communicate()
    if os.path.isfile(f'{trg_pth_dir}/{npy_file}.npy'):
        os.unlink(f'{trg_pth_dir}/{npy_file}.npy')
    if os.path.isfile(f'{origin_copy_pth}/{npy_file}.npy'):
        os.unlink(f'{origin_copy_pth}/{npy_file}.npy')
    
if __name__=='__main__':
    
    model_dic = {
                 '1': 'cubism_sd',
                 '2': 'cubism_anime',
                 '3': 'cubism_digital',
                 '4': 'cubism_impressionism',
                 '5': 'cubism_portrait',
                 '6': 'cubism_realistic',
                  
            }
    instance_style = { '1' : 'mid-journey', 
                      '2' : 'anime',
                      '3' : 'realistic',
                      '4' : 'romanticism',
                      '5' : 'impressionism',
                      '6' : 'pixel_art',
                      '7' : 'cubism',
                      }
    
    Comparative_group_one = True
    Comparative_group_two = False
    if Comparative_group_one and Comparative_group_two == False:
        for i in range(7,7+1):
            style = instance_style[str(i)]
            for j in range(1,6+1):
                model = model_dic[str(j)]    
                KID_score(model, style)
    
    elif Comparative_group_one==False and Comparative_group_two == True :
        for i in range(1, 3+1):
            style = instance_style[str(i)]
            
            model = model_dic[str(7)]
            KID_score(model, style)