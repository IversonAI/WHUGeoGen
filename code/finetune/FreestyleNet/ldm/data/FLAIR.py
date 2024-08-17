import os
from re import L
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# import blobfile as bf
from osgeo import gdal
import random

# Some words may differ from the class names defined in FLAIR to minimize ambiguity
FLAIR_dict = {

    # building	1	8.14	8.6
    # pervious surface	2	8.25	7.34
    # impervious surface	3	13.72	14.98
    # bare soil	4	3.47	4.36
    # water	5	4.88	5.98
    # coniferous	6	2.74	2.39
    # deciduous	7	15.38	13.91
    # brushwood	8	6.95	6.91
    # vineyard	9	3.13	3.87
    # herbaceous vegetation	10	17.84	22.17
    # agricultural land	11	10.98	6.95
    # plowed land  12

    # swimming pool	13	0.03	0.04
    # snow	14	0.15	-
    # clear cut	15	0.15	0.01
    # mixed	16	0.05	-
    # ligneous	17	0.01	0.03
    # greenhouse	18	0.12	0.2
    # other	19	0.14	-
    '1': 'building',  # 1 tokens
    '2': 'pervious surface',  # 4 tokens
    '3': 'impervious surface',  # 4 tokens
    '4': 'bare soil',  # 2 tokens
    '5': 'water',  # 1 tokens
    '6': 'coniferous',  # 3 tokens
    '7': 'deciduous',  # 3 tokens
    '8': 'brushwood',  # 2 tokens
    '9': 'vineyard',  # 2 tokens
    '10': 'herbaceous vegetation',  # 4 tokens
    '11': 'agricultural land',  # 4 tokens
    '12': 'plowed land',  # 3 tokens
    '13': 'swimming pool',# 3 tokens
    '14':'snow',# 1 tokens
    '15':'clear cut',# 2 tokens
    '16':'mixed',# 1 tokens
    '17':'ligneous', # 2 tokens
    '18':'greenhouse', # 2 tokens
    '19':'other',# 1 tokens
}


class FLAIRBase(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_p = flip_p

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path = self.image_paths[i]
        # pil_image = Image.open(path)
        # if not pil_image.mode == "RGB":
        #     pil_image = pil_image.convert("RGB")

        tifdata = gdal.Open(path)  # read tif
        # num_bands=tifdata.RasterCount #get bands
        # print(num_bands)
        tmp_img = tifdata.ReadAsArray()  # turn to array
        img_rgb = tmp_img.transpose(1, 2, 0)  # bands row col to row col bands
        # img_rgb=np.array(img_rgb,dtype=np.uint8) #set dtype=uint8

        # flair tif has 5 channels:red,blue,green,NIR,elevation

        pil_image = img_rgb[:, :, :3]
        pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")

        path_ = self.image_paths[i][:-4]
        # print('path_',path_)
        # print(path_.split('/')[-4:])
        tmp = path_.split('/')[-4:]
        TMP = tmp[3].split('_')[-1]
        # print('/'.join(path_.split('/')[-4:]))
        if 'train' in path_:
            # print(path_.split('/')[-1])
            path2 = os.path.join(self.data_root, 'labels/train/',
                                 tmp[0] + '/' + tmp[1] + '/msk/' + 'MSK' + '_' + TMP + '.tif')
        else:
            path2 = os.path.join(self.data_root, 'labels/val/',
                                 tmp[0] + '/' + tmp[1] + '/msk/' + 'MSK' + '_' + TMP + '.tif')
        pil_image2 = Image.open(path2)
        pil_image2 = pil_image2.convert("L")
        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            pil_image2 = pil_image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(pil_image2).astype(np.float32)
        # print('label',label)
        example["label"] = label

        if np.max(label) >= 20:
            print(path2)
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        # print('class_ids',class_ids)
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(20)
        # print('class_ids_final',class_ids_final)

        text = ''
        for i in range(len(class_ids)):
            text += FLAIR_dict[str(class_ids[i])]
            # print(text)
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        # print('class_ids_final-----',class_ids_final)
        example["class_ids"] = class_ids_final

        return example


class FLAIRTrain(FLAIRBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FLAIRValidation(FLAIRBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
