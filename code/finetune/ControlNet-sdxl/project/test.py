#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('..')

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from pytorch_lightning import seed_everything
from tqdm import tqdm

from cldm.model import create_model, load_state_dict
from dataset import WhuGeoGenDataset
from inference import run_sampler
from share import *
import os
# from PIL import Image
from glob import glob


def join(png1, png2, png3, size, flag='horizontal'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    # img1, img2 = Image.open(png1), Image.open(png2)
    # 统一图片尺寸，可以自定义设置（宽，高）
    # img1, img2, img3 = png1, png2, png3
    img1 = png1.resize((size, size), Image.NEAREST)
    img2 = png2.resize((size, size), Image.NEAREST)
    img3 = png3.resize((size, size), Image.NEAREST)

    size1, size2, size3 = img1.size, img2.size, img3.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0] + size2[0] + size3[0], size1[1]))
        loc1, loc2, loc3 = (0, 0), (size1[0], 0), (size1[0] + size2[0], 0)

        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.paste(img3, loc3)
    # joint.save('./results/multi_gpu/images/test_fake_imgs/showresults/{}.png'.format())
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
    # joint.save('./results/multi_gpu/images/test_fake_imgs/showresults/{}.png'.format())
    return joint


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = create_model('../models/cldm_v15.yaml').cpu()
model.load_state_dict(
    load_state_dict(
        "./experiments/lr=1e-05_bs=72/lightning_logs/train_UniOpenWithNYS16_fine_256_Nearest_6epoch_14h_0327/checkpoints/train_UniOpenWithNYS16_fine_256_Nearest_6epoch_14h_0327_epoch=6-step=8609.ckpt",
        location='cuda'))
model = model.cuda()

# # Running inference on the single test set

data_dir = "/home/user/dailei/Code_4090/ControlNet/training"
data_name = "NYS16"
split = "val"
category = "rgb_fine"
RESOLUTION = 256
dataset = WhuGeoGenDataset(data_dir=data_dir, data_name=data_name, split=split,
                           category=category,load_size=1024,crop_size=RESOLUTION)


#
# example = dataset[0]
# prompt = example["txt"]
# seg=example["hint"].astype(np.uint8)
# # sketch = dataset.sketches[0]
#
# print(prompt)
# print(seg)
# # seg=Image.fromarray(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB), "RGB")
#

#
# results = run_sampler(model, seg, prompt, seed=42)
#
# img=Image.fromarray(results[0], "RGB")
# img.save("111.jpg")
# print('iiiii')


# # Running inference on the whole test set



OUTPUT_DIR = f"whugeogen_{split}_outputs_{RESOLUTION}1/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in tqdm(range(len(dataset))):
    example = dataset[i]
    prompt = example["txt"]
    seg = example["hint"]
    name = "results_" + example["name"]+'.png'
    # sketch = cv2.imread(dataset.sketches[i])
    results = run_sampler(model, seg, prompt, image_resolution=RESOLUTION, seed=42, show_progress=False)
    # samples_image=results[0]
    samples_image = Image.fromarray(results[0], "RGB")
    source_seg = Image.fromarray(example["source_seg"],"RGB")
    target_img = Image.fromarray(example["target_img"],"RGB")

    # merge into an image
    joint = join(source_seg, target_img, samples_image, RESOLUTION, flag='horizontal')

    # add title
    saved = Image.new("RGB", (RESOLUTION * 3, RESOLUTION + 60))
    draw = ImageDraw.Draw(saved)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12, encoding="unic")
    # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24, encoding="unic")

    draw.text((0, 0), f"prompt:{prompt}", (255, 255, 255), font=font)
    draw.text((0, RESOLUTION + 30), f"Semantic Image", (255, 255, 255), font=font)
    draw.text((RESOLUTION, RESOLUTION + 30), f"RS Image", (255, 255, 255), font=font)
    draw.text((RESOLUTION * 2, RESOLUTION + 30), f"Generated Image", (255, 255, 255), font=font)

    bw, bh = saved.size
    lw, lh = joint.size
    saved.paste(joint, (bw - lw, int((bh - lh) / 2)))

    saved.save(os.path.join(OUTPUT_DIR, name))

#
# # ## Run inference for saved checkpoints for all hyperparams
#
# # In[ ]:
#
#
# import fnmatch
# import os
#
# experiments = [
#     'lr=1e-05_bs=2',
#     'lr=1e-05_bs=4',
#     'lr=5e-05_bs=2',
#     'lr=5e-05_bs=4',
#     'lr=5e-06_bs=2',
#     'lr=5e-06_bs=4',
# ]
#
# def load_model(model_path):
#     print("Loading pretrained model...")
#     model = create_model('../models/cldm_v15.yaml').cpu()
#     print("Loading model weights from: ", model_path)
#     model.load_state_dict(load_state_dict(model_path, location='cuda'))
#     model = model.cuda()
#     print("Done.")
#     return model
#
# def run_generate(model, dataset, out_dir, resolution=512):
#     os.makedirs(out_dir, exist_ok=True)
#
#     print("Writing generations to: ", out_dir)
#
#     for i in tqdm(range(len(dataset))):
#         example = dataset[i]
#         prompt = example["txt"]
#         sketch = cv2.imread(dataset.sketches[i])
#         results = run_sampler(model, sketch, prompt, image_resolution=resolution, seed=42, show_progress=False)
#
#         image = Image.fromarray(results[0], "RGB")
#         image.save(os.path.join(out_dir, f"image_{i:03d}.jpg"))
#
# def run_inference(exp,
#                   exp_dir='/raid/lingo/alexisro/ControlNet/project/experiments',
#                   gen_dir='/raid/lingo/alexisro/ControlNet/project/generations'):
#     """
#     exp_dir: points to where experiment subdirectories are
#     gen_dir: where to output generations
#     """
#
#     sub_exp_dir = os.path.join(exp_dir, exp)
#     model_paths = []
#
#
#     # TODO: what if there are multiple versions?
#     for dirpath, dirnames, filenames in os.walk(sub_exp_dir):
#         for filename in fnmatch.filter(filenames, '*.ckpt'):
#             full_path = os.path.join(dirpath, filename)
#             version = dirpath.split("/")[-2]
#             model_paths.append((full_path, version, filename))
#
#     for model_path, version, stub in model_paths:
#         out_dir = os.path.join(gen_dir, exp)
#         os.makedirs(out_dir, exist_ok=True)
#         log_txt_path = os.path.join(out_dir, 'model_path.txt')
#         print("Writing log txt to:", log_txt_path)
#         model = load_model(model_path)
#         with open(log_txt_path, 'w') as f:
#             print(model_path, file=f)
#         run_generate(model, dataset, out_dir)
#
# for exp_idx, exp in enumerate(experiments):
#     print("\n====================================================================================")
#     print(f"Running inference for experiment {exp_idx+1}/{len(experiments)}:", exp)
#     run_inference(exp)
#
#
