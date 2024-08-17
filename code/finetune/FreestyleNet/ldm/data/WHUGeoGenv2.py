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
import json

mapping2rgb = {
    # 地形分类（细）
    # 'railway': (0, 139, 139),
    # 'motorway': (184, 134, 11),
    # 'ordinary highway': (169, 169, 169),
    # 'river': (0, 100, 0),
    # 'lake': (72, 61, 139),
    # 'reservoir': (189, 183, 107),
    # 'sea': (139, 0, 139),
    # 'island': (85, 107, 47),
    # 'swamp': (255, 140, 0),
    # 'farmland': (153, 50, 204),
    # 'forest': (139, 0, 0),
    # 'urban green': (233, 150, 122),
    # 'meadow': (143, 188, 143),
    # 'space': (47, 79, 79),
    # 'buildings': (119, 136, 153),
    # 'port': (0, 206, 209),
    # 'parking lot': (148, 0, 211),
    # 'airport': (255, 20, 147),
    # 'overpass': (0, 191, 255),
    # 'bridge': (105, 105, 105),
    # 'railway station': (176, 196, 222),
    # 'crossroads': (30, 144, 255),
    # 'roundabout': (178, 34, 34),
    # 'heliport': (255, 250, 240),
    # 'terminal': (34, 139, 34),
    # 'lighthouse': (255, 0, 255),
    # 'fuel': (220, 220, 220),
    # 'solar panel': (248, 248, 255),
    # 'generator': (255, 215, 0),
    # 'electricity tower': (218, 165, 32),
    # 'communications tower': (128, 128, 128),
    # 'school': (0, 128, 0),
    # 'sport field': (173, 255, 47),
    # 'water tower': (255, 255, 224),
    # 'amusement park': (240, 255, 240),
    # 'park': (255, 105, 180),
    # 'grave yard': (205, 92, 92),
    # 'hospital': (75, 0, 130),
    # 'church': (0, 255, 0),
    # 'palace': (240, 230, 140),
    # 'fire station': (230, 230, 250),
    # 'police station': (255, 240, 245),
    # 'prison': (124, 252, 0),
    # 'tank': (255, 250, 205),
    # 'bare land': (173, 216, 230),
    # 'sand': (240, 128, 128),
    # 'glacier': (224, 255, 255),
    # 'mountain': (250, 250, 210),
    # 'snowfield': (211, 211, 211),
    # 'plane': (144, 238, 144),
    # 'vehicle': (255, 182, 193),
    # 'ship': (255, 160, 122),
    # 'person': (32, 178, 170),
    # 'animal': (135, 206, 250)

    (0, 139, 139): 1,
    (184, 134, 11): 2,
    (169, 169, 169): 3,
    (0, 100, 0): 4,
    (72, 61, 139): 5,
    (189, 183, 107): 6,
    (139, 0, 139): 7,
    (85, 107, 47): 8,
    (255, 140, 0): 9,
    (153, 50, 204): 10,
    (139, 0, 0): 11,
    (233, 150, 122): 12,
    (143, 188, 143): 13,
    (47, 79, 79): 14,
    (119, 136, 153): 15,
    (0, 206, 209): 16,
    (148, 0, 211): 17,
    (255, 20, 147): 18,
    (0, 191, 255): 19,
    (105, 105, 105): 20,
    (176, 196, 222): 21,
    (30, 144, 255): 22,
    (178, 34, 34): 23,
    (255, 250, 240): 24,
    (34, 139, 34): 25,
    (255, 0, 255): 26,
    (220, 220, 220): 27,
    (248, 248, 255): 28,
    (255, 215, 0): 29,
    (218, 165, 32): 30,
    (128, 128, 128): 31,
    (0, 128, 0): 32,
    (173, 255, 47): 33,
    (255, 255, 224): 34,
    (240, 255, 240): 35,
    (255, 105, 180): 36,
    (205, 92, 92): 37,
    (75, 0, 130): 38,
    (0, 255, 0): 39,
    (240, 230, 140): 40,
    (230, 230, 250): 41,
    (255, 240, 245): 42,
    (124, 252, 0): 43,
    (255, 250, 205): 44,
    (173, 216, 230): 45,
    (240, 128, 128): 46,
    (224, 255, 255): 47,
    (250, 250, 210): 48,
    (211, 211, 211): 49,
    (144, 238, 144): 50,
    (255, 182, 193): 51,
    (255, 160, 122): 52,
    (32, 178, 170): 53,
    (135, 206, 250): 54,
}
# Some words may differ from the class names defined in WHUGeoGenv2 to minimize ambiguity
WHUGeoGenv2_dict = {

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
    # '1': 'building',  # 1 tokens
    # '2': 'pervious surface',  # 4 tokens
    # '3': 'impervious surface',  # 4 tokens
    # '4': 'bare soil',  # 2 tokens
    # '5': 'water',  # 1 tokens
    # '6': 'coniferous',  # 3 tokens
    # '7': 'deciduous',  # 3 tokens
    # '8': 'brushwood',  # 2 tokens
    # '9': 'vineyard',  # 2 tokens
    # '10': 'herbaceous vegetation',  # 4 tokens
    # '11': 'agricultural land',  # 4 tokens
    # '12': 'plowed land',  # 3 tokens
    # '13': 'swimming pool',# 3 tokens
    # '14':'snow',# 1 tokens
    # '15':'clear cut',# 2 tokens
    # '16':'mixed',# 1 tokens
    # '17':'ligneous', # 2 tokens
    # '18':'greenhouse', # 2 tokens
    # '19':'other',# 1 tokens

    '1': 'railway',
    '2': 'motorway',
    '3': 'ordinary highway',
    '4': 'river',
    '5': 'lake',
    '6': 'reservoir',
    '7': 'sea',
    '8': 'island',
    '9': 'swamp',
    '10': 'farmland',
    '11': 'forest',
    '12': 'urban green',
    '13': 'meadow',
    '14': 'space',
    '15': 'buildings',
    '16': 'port',
    '17': 'parking lot',
    '18': 'airport',
    '19': 'overpass',
    '20': 'bridge',
    '21': 'railway station',
    '22': 'crossroads',
    '23': 'roundabout',
    '24': 'heliport',
    '25': 'terminal',
    '26': 'lighthouse',
    '27': 'fuel',
    '28': 'solar panel',
    '29': 'generator',
    '30': 'electricity',
    '31': 'communication tower',
    '32': 'school',
    '33': 'sport field',
    '34': 'water tower',
    '35': 'amusement park',
    '36': 'park',
    '37': 'grave yard',
    '38': 'hostpital',
    '39': 'church',
    '40': 'palace',
    '41': 'fire station',
    '42': 'police station',
    '43': 'prison',
    '44': 'tank',
    '45': 'bare land',
    '46': 'sand',
    '47': 'glacier',
    '48': 'mountain',
    '49': 'snowfield',
    '50': 'plane',
    '51': 'vehicle',
    '52': 'ship',
    '53': 'person',
    '54': 'animal',

}


class WHUGeoGenv2Base(Dataset):
    # def __init__(self, data_dir="/data/dailei/WHUGeoGen3", split="finetune", res_name="data512",
    #              data_name="BJ_NY", crop_size=512):
    #     self.data = []
    #     self.data_dir = data_dir
    #     self.data_name = data_name
    #     self.res_name = res_name
    #     self.crop_size = crop_size
    #     self.split = split
    #
    #     with open(os.path.join(self.data_dir, split, f'{res_name}_{data_name}.json'), 'rt') as f:
    #         for line in f:
    #             self.data.append(json.loads(line))
    #
    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     example=dict()
    #     source_filename = item['source']
    #     target_filename = item['target']
    #     prompt = item['prompt']
    #     name = (target_filename.split('/')[-1])[:-4]
    #     # print(name)
    #
    #     source_img = Image.open(os.path.join(self.data_dir, source_filename)).convert('RGB')
    #     target = Image.open(os.path.join(self.data_dir, target_filename)).convert('RGB')
    #     target, source = self.transforms(target, source_img, source_img.size)
    #     # print(crop_coords_top_left)
    #     source_seg = np.array(source)
    #     target_img = np.array(target)
    #
    #     source = source_seg
    #     target = target_img
    #
    #     # convert channel from HWC to CHW
    #     # source = einops.rearrange(source, 'h w c -> c h w')
    #     # target = einops.rearrange(target, 'h w c -> c h w')
    #
    #     # Normalize source images to [0, 1].
    #     # source = np.float32(source) / 255.0
    #
    #     # Normalize target images to [-1, 1].
    #     target = (np.float32(target) / 127.5) - 1.0
    #
    #
    #     if np.max(source) >= 55:
    #         print(name)
    #
    #     example["image"] =target
    #     label = np.array(source).astype(np.float32)
    #     # print('label',label)
    #     example["label"] = label
    #
    #     class_ids = sorted(np.unique(source.astype(np.uint8)))
    #     # print('class_ids',class_ids)
    #     if class_ids[0] == 0:
    #         class_ids = class_ids[1:]
    #     class_ids_final = np.zeros(55)
    #     # print('class_ids_final',class_ids_final)
    #
    #     text = ''
    #     for i in range(len(class_ids)):
    #         text += WHUGeoGenv2_dict[str(class_ids[i])]
    #         # print(text)
    #         text += ' '
    #         class_ids_final[class_ids[i]] = 1
    #     text = text[:-1]
    #     example["caption"] = text
    #     # print('class_ids_final-----',class_ids_final)
    #     example["class_ids"] = class_ids_final
    #
    #     return example
    def __init__(self,
                 data_root,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.data_paths = txt_file
        self.image_paths = []
        with open(self.data_paths, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for i in range(len(lines)):
                if 'low' in lines[i]:
                    self.image_paths.append(lines[i])
                elif 'mid' in lines[i]:
                    if i % 2 == 0:
                        self.image_paths.append(lines[i])
                elif 'high' in lines[i]:
                    if i % 10 == 0:
                        self.image_paths.append(lines[i])
            # if 'high' in
            # self.image_paths = f.read().splitlines()

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
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        # tifdata = gdal.Open(path)  # read tif
        # # num_bands=tifdata.RasterCount #get bands
        # # print(num_bands)
        # tmp_img = tifdata.ReadAsArray()  # turn to array
        # img_rgb = tmp_img.transpose(1, 2, 0)  # bands row col to row col bands
        # # img_rgb=np.array(img_rgb,dtype=np.uint8) #set dtype=uint8
        #
        # # flair tif has 5 channels:red,blue,green,NIR,elevation
        #
        # pil_image = img_rgb[:, :, :3]
        # pil_image = Image.fromarray(pil_image)
        # pil_image = pil_image.convert("RGB")

        # path_ = self.image_paths[i][:-4]
        # print('path_',path_)
        # print(path_.split('/')[-4:])
        # tmp = path_.split('/')
        # print(tmp)
        # / data / dailei / WHUGeoGen3 / finetune / data512 / low / RS_images / BeiJing / BJ1/...jpg
        # TMP = tmp[3].split('_')[-1]
        # print('/'.join(path_.split('/')[-4:]))
        if 'finetune' in path:
            # print(path_.split('/')[-1])
            path2 = path
        else:
            path2 = path.replace('finetune', 'test')
            path2 = path2.replace('RS_images', 'Semantic_masks')
            path2 = path2.replace('jpg', 'png')
            # print(path2)
        pil_image2 = Image.open(path2)
        # 将图像转换为RGB模式，确保颜色通道是正确的
        rgb_image = pil_image2.convert('RGB')
        # 创建一个新的灰度图像
        gray_image = Image.new('L', rgb_image.size)
        # 获取RGB图像的像素数据
        pixels = rgb_image.load()
        # 遍历图像中的每个像素
        for y in range(rgb_image.size[1]):
            for x in range(rgb_image.size[0]):
                # 获取当前像素的RGB值
                r, g, b = pixels[x, y]
                # 将RGB元组转换为可以用于字典键的格式
                pixel_key = (r, g, b)
                # 如果当前RGB值在映射表中，则将对应的灰度值赋给灰度图像
                if pixel_key in mapping2rgb:
                    gray_image.putpixel((x, y), mapping2rgb[pixel_key])
                else:
                    # 如果不在映射表中，可以选择一个默认值或者保持原样
                    gray_image.putpixel((x, y), 0)

                    # pil_image2 = pil_image2.convert("L")
        flip = random.random() < self.flip_p
        if self.size is not None:
            pil_image = pil_image.resize((self.size, self.size), resample=self.interpolation)
            gray_image = gray_image.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        if flip:
            pil_image = pil_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            gray_image = gray_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(gray_image).astype(np.float32)

        # print('label',label)
        example["label"] = label

        if np.max(label) >= 55:
            print(path2)
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        # print('class_ids',class_ids)
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(55)
        # print('class_ids_final',class_ids_final)

        text = ''
        for i in range(len(class_ids)):
            text += WHUGeoGenv2_dict[str(class_ids[i])]
            # print(text)
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        # print('class_ids_final-----',class_ids_final)
        example["class_ids"] = class_ids_final

        return example


class WHUGeoGenv2Train(WHUGeoGenv2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class WHUGeoGenv2Validation(WHUGeoGenv2Base):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(flip_p=flip_p, **kwargs)
