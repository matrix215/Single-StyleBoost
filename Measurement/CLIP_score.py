import img_rename2, os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from PIL import ImageFile
def calculate_clip_score(images, prompts):
        #print(np.shape(images))
        images = images.astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)
def cmd_cal_clip_score(model, style, prompt_id):
    #변환할 이미지 목록 불러오기
    clip_score_lst = []
    for index in tqdm(prompt_id.keys(), desc="EVALUATE"):
        if len(prompt_id[str(index)]) > 77:
            continue
        image_path = f'./output_img/{model}/{style}/{index}'
        img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
        img_list_png = [img for img in img_list if img.endswith(".png")] #지정된 확장자만 필터링
        img_len = np.shape(img_list_png)[0]
        prompts = []
        for _ in range(img_len):
            prompts.append(prompt_id[index])
        img_list_png = np.array(img_list_png)
        img_list_np = []
        for i in img_list_png:
            img = Image.open(image_path + '/' + i)
            img_array = np.array(img)
            img_list_np.append(img_array)
        img_np = np.array(img_list_np)
        sd_clip_score = calculate_clip_score(img_np, prompts)
        clip_score_lst.append(sd_clip_score)
    return clip_score_lst
if __name__=="__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    prompt_id = dict()
    with open('./prompt.txt','r') as f:
        prompt_lst = f.readlines()
        for i in range(1,len(prompt_lst)+1):
            prompt_id[str(i)] = prompt_lst[i-1].rstrip('\n')
    model_dic = {'0': 'mid'
            }
    instance_style = { '1' : 'jouner-mid-1000',
                      '2' : 'anime',
                      '3' : 'realistic'}
    Comparative_group_one = True
    if Comparative_group_one:
        for i in range(1,1+1):
            style = instance_style[str(i)]
            for j in range(0,0+1):
                model = model_dic[str(j)]
                clip_score_lst = cmd_cal_clip_score(model, style, prompt_id)
                sum_of_score = sum(clip_score_lst)/len(clip_score_lst)
                all_imgs = os.listdir(f'./output_img/{model}/{style}')
                imgs = [file for file in all_imgs if not file.endswith(".npy")]
                total_len = 0
                for pth in imgs:
                    if not pth=='all_images':
                        file_path = f'./output_img/{model}/{style}/{pth}'
                        total_len = total_len + len(os.listdir(file_path))
                with open('clip_score.txt','a') as f:
                    now = datetime.now()
                    f.write(f'\ndate : {now.date()} {now.time()}\nmodel : {model}\t|style : {style}\t|score : {clip_score_lst}\t|sum of score : {sum_of_score}\
                        \n|# of image : {total_len}')