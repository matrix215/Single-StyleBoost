import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import subprocess 
import json
import torch 
import requests
#드림부스 다운
'''
subprocess.call('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py',shell=True)

subprocess.call('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py',shell=True)

subprocess.call('pip install -qq git+https://github.com/ShivamShrirao/diffusers',shell=True)

subprocess.call('pip install -q -U --pre triton',shell=True)

subprocess.call('pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers',shell=True)

#####허깅페이스 경로
path='/home/kkko/capston_design/.huggingface'
os.mkdir(path)
'''

def training_dreambooth(gpu, instance_prompt, model_dic, instance_style, class_img_dic):
    #GPU setting
    #torch.cuda.set_device(int(gpu))
    
    HUGGINGFACE_TOKEN = "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"
    os.system(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token')

    #base model
    #맨 처음 model weight를 가져오는 경로
    #MODEL_NAME = f"./civit_model_diffusers/{model}" #@param {type:"string"}
    #이후 model weight 가져 오는 경로
    #MODEL_NAME = f"/home/kkko/paper/style_transfer_paper/model_weight/multi_seper_ani_people_basic_750/anime_people2"
    
    #MODEL_NAME = f"runwayml/stable-diffusion-v1-5"
    MODEL_NAME = f"/home/kkko/paper/style_transfer_paper/model_weight/multi_seper_ani_back_basic_1000/anime_back_20"
    

    #weight가 저장될 경로  /home/kkko/paper/style_transfer_paper/model_weight
    OUTPUT_DIR = f"/home/kkko/paper/style_transfer_paper/model_weight/{model_dic}/{instance_style}"
    print(f"[*] Weights will be saved at {OUTPUT_DIR}")
    
    # You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
    concepts_list = [
        {
            "instance_prompt":      f"a {instance_prompt} person",
            "class_prompt":         "a photo of person",
            
            "instance_data_dir":    f"/home/kkko/paper/style_transfer_paper/instance_img/{instance_style}",
            "class_data_dir":       f"/home/kkko/paper/style_transfer_paper/class_img/{class_img_dic}"
        }
    ]

    # `class_data_dir` contains regularization images

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)
        
    #vae : stabilityai/sd-vae-ft-mse ,./anime_vae/anything
    subprocess.call(f'python3 train_dreambooth_ori.py \
    --pretrained_model_name_or_path={MODEL_NAME} \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir={OUTPUT_DIR} \
    --revision="fp16" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --seed=1337 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_text_encoder \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=20 \
    --sample_batch_size=1 \
    --max_train_steps=1000 \
    --save_interval=10000 \
    --concepts_list="/home/kkko/paper/style_transfer_paper/concepts_list.json"',shell=True)

    print("instance prompt : ", instance_prompt)
    

if __name__=='__main__':
    instance_prompt = 'ukj'
    model_dic = {'0' : 're_img_5_750_back',
                '1' : 're_img_5_750_people',
                '2' : 're_img_5_750_people_back', # training step 비교군 1
                '3' : 're_img_5_1000_back',
                '4' : 're_img_5_1000_people',
                
                
                '5' : 'face_t',
                '6' : 'test',
                '7' : 'in_img_face_with_background_2000',
                '8' : 'in_img_face_with_background_3000', 
                '9' : 'in_img_face_500_style_by_mid',
                '10' : 're_img_50_500_peopl_back',
                '11' : 're_img_50_500_back',
                '12' : 're_img_5_500_back',
                '13' : 're_img_5_500_people',
                '14' : 're_img_5_500_people_back',
                '15' : 'new_re_img_5_750_back',
                '16' : 'new_re_img_0_1000_back',
                '17' : 'new_re_img_0_750_back',
                '18' : 'new_re_img_8_500_back',
                '19' : 'new_re_img_0_500_back',
                '20' : 'new_re_img_8_1000_back',
                '21' : 'new_re_img_0_1200_back',
                '22' : 'new_re_img_8_1200_back',
                '23' : 'new_re_img_0_1500_back',
                '24' : 'new_re_img_8_1500_back',
                '25' : 'mutli_seper_ani_comp_2000',
                '26' : 'multi_seper_ani_back_basic_1000'
                
                
            }
    instance_style = { '1' : 'mid-journey', 
                      '2' : 'anime_back_20',
                      '3' : 'realistic',
                      '4' : 'realistic_back_20',
                      '5' : 'anime_people2'
                      }
    class_img_dic = {'1' : 'mid-journey',
                     '2' : 'anime',
                     '3' : 'realistic',
                     '4' : 'ani_people_back',#65
                     '5' : 'ani_back',
                     '6' : 're_people_back',
                     '7' : 'md_people_back',
                     '8' : 'test2',
                     '9' : 'digital_all',
                     '10' : 'digial_background',
                     '11' : 'digital_person',
                     '12' : 'digital_person3'
                     }  # 146
    
    training_dreambooth(3, instance_prompt, model_dic['25'], instance_style['5'], class_img_dic['12'])