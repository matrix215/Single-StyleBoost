import os, subprocess
os.environ['CUDA_VISIBLE_DEVICES']='3'
from os import path
import shutil

def FID_measurement(model,style):
    origin_pth_dir = f'/home/kkko/output_img/realistic' 
    trg_pth_dir = f'/home/kkko/paper/style_transfer_paper/output_img/{model}/{style}'
    copy_pth = f'/home/kkko/paper/style_transfer_paper/output_img/{model}/{style}/all_images'
    origin_copy_pth = f'/home/kkko/paper/style_transfer_paper/output_img/{style}/all_images'
    all_index_lst = os.listdir(trg_pth_dir)
    index_lst = [file for file in all_index_lst if not file.endswith(".npy")]
    
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
    print('33')
    all_index_lst = os.listdir(origin_pth_dir)        
    index_lst = [file for file in all_index_lst if not file.endswith(".npy")]
    print('335')
    print(len(index_lst))
    for index in index_lst:
        file_pth_dir = origin_pth_dir + '/' + index
        
        file_lst = os.listdir(file_pth_dir)
        #print(file_lst)
        

        if not path.isdir(origin_copy_pth):
            os.makedirs(origin_copy_pth)
       
        for file in file_lst:
            if not path.exists(origin_copy_pth + '/' + file):
                shutil.copy(file_pth_dir+'/'+file, origin_copy_pth+'/'+file)
                
    print('4444')
    proc = subprocess.Popen([f"python /home/kkko/paper/style_transfer_paper/PyTorch-FID-score/fid_score.py {origin_copy_pth} {copy_pth} -c 0 --model {model} --style {style}"],  shell=True)
    _ , _ = proc.communicate()
    copy_file_lst = os.listdir(copy_pth)
    with open('fid_score.txt','a') as f:
        f.write(f'\n# of image : {len(copy_file_lst)}')
    for file in copy_file_lst:
        if path.exists(copy_pth+'/'+file):
            os.unlink(copy_pth+'/'+file)

    copy_file_lst2 = os.listdir(origin_copy_pth)
    for file in copy_file_lst2:
        if path.exists(origin_copy_pth+'/'+file):
            os.unlink(origin_copy_pth+'/'+file)
    
if __name__=="__main__":
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model_dic = {'0' : 're_img_5_750_people_back',
                 '1' : 're_img_5_750_back',
                 '2' : 're_img_5_750_people',
                 '3' : 're_img_5_500_back',
                 '4' : 're_img_5_500_people',
                 '5' : 're_img_5_500_people_back',
                 '6' : 're_img_5_1000_back',
                 '7' : 're_img_5_1000_people_back',
                 '8' : 're_img_5_1000_people',
                 
                 '10': 'new_re_img_0_1000_back',
                 '11': 'new_re_img_5_750_back',
                 
                 '12': 'new_re_img_0_750_back',
                 '13': 'new_re_img_8_500_back',
                 '14': 'new_re_img_0_500_back',
                 '15': 'new_re_img_8_1000_back',
                 '16': 'new_re_img_0_1200_back',
                 '17': 'new_re_img_8_1200_back',
                 '18': 'new_re_img_0_1500_back',
                 '19': 'new_re_img_8_1500_back',
                 '20': 'mutli_thrid_re_2_2000'
                 
                
                
            }
    instance_style = { '1' : 'mid-journey', 
                      '2' : 'anime',
                      '3' : 'realistic'}
    Comparative_group_one = True
    Comparative_group_two = False
    if Comparative_group_one and Comparative_group_two == False:
        for i in range(3,3+1):
            style = instance_style[str(i)]
            for j in range(20,20+1): #range(1,3+1):
                model = model_dic[str(j)]    
                FID_measurement(model, style)
    
    elif Comparative_group_one==False and Comparative_group_two == True :
        for i in range(2, 2+1):
            style = instance_style[str(i)]
            
            model = model_dic[str(20)]
            FID_measurement(model, style)
        