import os
import json
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from ast import literal_eval
from torchvision import transforms as TR
from PIL import Image
import torch
import einops

class WhuGeoGenDataset(Dataset):
    def __init__(self, data_dir="/home/user/dailei/Code_4090/ControlNet/training", data_name="NYS16", split="train",
                 category="rgb_fine", load_size=1024, crop_size=256):
        self.data = []
        self.data_dir = data_dir
        self.data_name = data_name
        self.split = split
        self.category = category
        self.load_size = load_size
        self.crop_size = crop_size
        if data_name == "UniOpen":
            with open(os.path.join(self.data_dir, f'{category}_{split}.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif data_name == "UniOpenWithNYS16":
            with open(os.path.join(self.data_dir, f'{category}_{split}_with_nys16.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            with open(os.path.join(self.data_dir, data_name, category,
                                   f'blip2_caption_controlnet_{category}_seg_{data_name}_{split}.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        name = (target_filename.split('/')[-1])[:-4]
        # print(name)

        if self.data_name == "UniOpen" or self.data_name == "UniOpenWithNYS16":
            # source = cv2.imread(os.path.join(self.data_dir, source_filename), cv2.IMREAD_COLOR)
            # target = cv2.imread(os.path.join(self.data_dir, target_filename), cv2.IMREAD_COLOR)

            source = Image.open(os.path.join(self.data_dir, source_filename)).convert('RGB')
            target = Image.open(os.path.join(self.data_dir, target_filename)).convert('RGB')
        else:
            source = Image.open(
                os.path.join(self.data_dir, self.data_name, self.category, self.split, source_filename)).convert('RGB')
            target = Image.open(
                os.path.join(self.data_dir, self.data_name, self.category, self.split, target_filename)).convert('RGB')
            # source = cv2.imread(
            #     os.path.join(self.data_dir, self.data_name, self.category, self.split, source_filename),
            #     cv2.IMREAD_COLOR)
            # target = cv2.imread(
            #     os.path.join(self.data_dir, self.data_name, self.category, self.split, target_filename),
            #     cv2.IMREAD_COLOR)

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # INTER_LINEAR：双线性插值，会改变颜色
        # INTER_NEAREST：最近邻插值，不会改变
        # source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_NEAREST)# 3-25 changed to INTER_NEAREST
        # target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST)

        # source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)  # 3-25 changed to INTER_NEAREST
        # target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)

        target, source = self.transforms(target, source)

        source_seg = np.array(source)
        target_img = np.array(target)

        source = source_seg
        target = target_img
        # print("-----------------------")
        # print(type(source))
        # print("-----------------------")

        # # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0
        #
        # # Normalize target images to [-1, 1].
        # target = (target.astype(np.float32) / 127.5) - 1.0

        # Normalize source images to [0, 1].
        source = np.float32(source) / 255.0

        # Normalize target images to [-1, 1].
        target = (np.float32(target) / 127.5) - 1.0
        return dict(jpg=target, txt=prompt, hint=source, name=name, source_seg=source_seg, target_img=target_img)

    def transforms(self, image, label):
        # assert image.size == label.size
        # resize
        new_width, new_height = (self.load_size, self.load_size)
        image = TR.functional.resize(image, (new_width, new_height), TR.transforms.InterpolationMode.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), TR.transforms.InterpolationMode.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        # flip
        # if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
        if random.random() < 0.5:
            image = TR.functional.hflip(image)
            label = TR.functional.hflip(label)
        # to tensor
        # image = TR.functional.to_tensor(image)
        # label = TR.functional.to_tensor(label)
        # normalize
        # image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
class WhuGeoGenv2Dataset(Dataset):
    def __init__(self, data_dir="/data/dailei/WHUGeoGen3", split="finetune",res_name="data512",
                 data_name="BJ_NY", crop_size=512):
        self.data = []
        self.data_dir = data_dir
        self.data_name = data_name
        self.res_name = res_name
        self.crop_size = crop_size
        self.split = split
        if res_name == "Airport" or res_name == "Port" or res_name == "Tank":
            with open(os.path.join(self.data_dir, split, f'{res_name}.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))

        else:
            with open(os.path.join(self.data_dir,split, f'{res_name}_{data_name}.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        name = (target_filename.split('/')[-1])[:-4]
        # print(name)

        source_img = Image.open(os.path.join(self.data_dir, source_filename)).convert('RGB')
        target = Image.open(os.path.join(self.data_dir, target_filename)).convert('RGB')
        target, source = self.transforms(target, source_img,source_img.size)
        # print(crop_coords_top_left)
        source_seg = np.array(source)
        target_img = np.array(target)

        source = source_seg
        target = target_img

        # convert channel from HWC to CHW
        # source = einops.rearrange(source, 'h w c -> c h w')
        # target = einops.rearrange(target, 'h w c -> c h w')

        # Normalize source images to [0, 1].
        source = np.float32(source) / 255.0

        # Normalize target images to [-1, 1].
        target = (np.float32(target) / 127.5) - 1.0
        # target_norm = target.astype(np.float32) / 255.0
        # target = 2.0 * target_norm - 1.0
        return dict(jpg=target, txt=prompt, hint=source, name=name, source_seg=source_seg, target_img=target_img)

    def transforms(self, image, label,size):
        assert image.size == label.size
        # resize
        new_width, new_height = (size[0], size[1])
        image = TR.functional.resize(image, (new_width, new_height), TR.transforms.InterpolationMode.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), TR.transforms.InterpolationMode.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
        # flip
        # if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
        if not self.split == "test":
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        # image = TR.functional.to_tensor(image)
        # label = TR.functional.to_tensor(label)
        # normalize
        # image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label

class Fill50kDataset(Dataset):
    def __init__(self, split="train"):
        self.data = []
        self.data_dir = './data/fill50k'
        with open(os.path.join(self.data_dir, f'{split}.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.data_dir, 'fill50k/' + source_filename))
        target = cv2.imread(os.path.join(self.data_dir, 'fill50k/' + target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # source = cv2.resize(source, (256, 256),interpolation=cv2.INTER_LANCZOS4)
        # target = cv2.resize(target, (256, 256),interpolation=cv2.INTER_NEAREST)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class PhotoSketchDataset(Dataset):
    def __init__(self, split="train", data_dir="./data/sketch"):
        self.sketches = []
        self.images = []
        self.prompts = []

        self.data_dir = data_dir
        self.sketch_dir = os.path.join(self.data_dir, 'sketch-rendered/width-5')
        self.img_dir = os.path.join(self.data_dir, 'image')

        self.padded_sketch_dir = os.path.join(self.sketch_dir, 'padded')
        self.padded_img_dir = os.path.join(self.img_dir, 'padded')

        os.makedirs(self.padded_sketch_dir, exist_ok=True)
        os.makedirs(self.padded_img_dir, exist_ok=True)

        # read in splits
        with open(os.path.join(self.data_dir, f'list/{split}.txt'), 'rt') as f:
            for line in f:
                img_idx = line.strip()
                for idx in range(1, 6):
                    sketch_stub = f'{img_idx}_0{idx}.png'
                    img_stub = f'{img_idx}.jpg'

                    prompt_path = os.path.join(f'./lavis/captions/{img_idx}.json')
                    self.prompts.append(prompt_path)

                    # source is the sketch
                    source = cv2.imread(os.path.join(self.sketch_dir, sketch_stub))
                    target = cv2.imread(os.path.join(self.img_dir, img_stub))

                    source, target = self.process_imgs(source, target)

                    sketch_path = os.path.join(self.padded_sketch_dir, sketch_stub)
                    cv2.imwrite(sketch_path, source)
                    # add the padded sketch path
                    self.sketches.append(sketch_path)

                    img_path = os.path.join(self.padded_img_dir, img_stub)
                    cv2.imwrite(img_path, target)
                    # add the padded img path
                    self.images.append(img_path)

        assert len(self.prompts) == len(self.sketches)
        assert len(self.images) == len(self.sketches)

    def __len__(self):
        return len(self.sketches)

    # Crop images to 512 x 512, then pad
    # TODO: instead of cropping, rescale?

    def process_imgs(self, source, target):

        max_height = 512
        max_width = 512

        source = source[:max_height, :max_width, :]
        target = target[:max_height, :max_width, :]

        source_height, source_width, _ = source.shape
        target_height, target_width, _ = target.shape

        # pad
        source = cv2.copyMakeBorder(source, 0, max_height - source_height, 0, max_width - source_width,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
        target = cv2.copyMakeBorder(target, 0, max_height - target_height, 0, max_width - target_width,
                                    borderType=cv2.BORDER_CONSTANT, value=0)

        return source, target

    def __getitem__(self, idx):
        # assuming that we get the items by looping through sketch 1 of all images, then sketch 2 of all images, etc.

        prompt_file = self.prompts[idx]
        with open(prompt_file) as f:
            prompt = literal_eval(f.read().strip())[0]

        # source is the sketch
        source = cv2.imread(self.sketches[idx])
        target = cv2.imread(self.images[idx])

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        assert source.shape == target.shape

        return dict(jpg=target, txt=prompt, hint=source)
