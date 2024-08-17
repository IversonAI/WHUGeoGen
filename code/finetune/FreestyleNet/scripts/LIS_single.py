import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.COCO import COCO_dict
from ldm.data.ADE20K import ADE20K_dict
import random

from torch.utils.data import DataLoader, Dataset
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

class COCOVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
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

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'stuffthingmaps_trainval2017/val2017', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[-1] == 255:
            class_ids = class_ids[:-1]
        class_ids_final = np.zeros(182)
        text = ''
        for i in range(len(class_ids)):
            text += COCO_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class ADE20KVal(Dataset):
    def __init__(self,
                 data_root,
                 txt_file,
                 size=512,
                 interpolation="bicubic",
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

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        path_ = self.image_paths[i][:-4]
        path2 = os.path.join(self.data_root, 'annotations/validation', path_.split('/')[-1] + '.png')
        pil_image2 = Image.open(path2)
        example["img_name"] = path_.split('/')[-1]

        pil_image2 = pil_image2.resize((self.size, self.size), resample=PIL.Image.NEAREST)
        label = np.array(pil_image2).astype(np.float32)
        example["label"] = label
        class_ids = sorted(np.unique(label.astype(np.uint8)))
        if class_ids[0] == 0:
            class_ids = class_ids[1:]
        class_ids_final = np.zeros(151)
        text = ''
        for i in range(len(class_ids)):
            text += ADE20K_dict[str(class_ids[i])]
            text += ' '
            class_ids_final[class_ids[i]] = 1
        text = text[:-1]
        example["caption"] = text
        example["class_ids"] = class_ids_final

        return example

class WHUGeoGenv2Val(Dataset):
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
        for root_dir, sub_dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith(".jpg"):
                    img = os.path.join(root_dir, file)
                    self.image_paths.append(img)




            # for i in range(len(lines)):
            #     if 'low' in lines[i]:
            #         self.image_paths.append(lines[i])
            #     elif 'mid' in lines[i]:
            #         if i % 20 == 0:
            #             self.image_paths.append(lines[i])
            #     elif 'high' in lines[i]:
            #         if i % 50 == 0:
            #             self.image_paths.append(lines[i])
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
        print(path)
        seg = path.replace('.jpg', '.png')
        caption = path.replace('.jpg', '.caption')
        with open(caption, 'r') as f1:
            caption = f1.readline().strip()
        pil_image = Image.open(path)
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")

        tmp = seg.split('/')[-1]
        example["img_name"] = tmp
        # print(example["img_name"])
        pil_image2 = Image.open(seg)
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
        # image = np.array(pil_image).astype(np.uint8)
        # example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        label = np.array(gray_image).astype(np.float32)

        width2, height2 = gray_image.size
        # # # 创建一个空白的图像，大小与原图像相同
        # # blank_image = Image.new('RGB', (width2, height2), (255, 255, 255))  # 白色背景
        # # blank_image=np.array(blank_image).astype(np.float32)
        # #blank_image = np.ones((height2, width2, 3), dtype=np.uint8) * 255

        # print('label',label)
        example["label"] = label
        # example["label"] =np.full((width2, height2), 255, dtype=np.uint8)
        print(label)
        if np.max(label) >= 55:
            print(seg)
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
        example["caption"] = caption
        # print('class_ids_final-----',class_ids_final)
        example["class_ids"] = class_ids_final

        return example
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/layout2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-finetune_COCO.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True, 
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="path to txt file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset to evaluate",
        choices=["COCO", "ADE20K", "WHUGeoGenv2"],
        default="COCO"
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size
    if opt.dataset == "COCO":
        val_dataset = COCOVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "ADE20K":
        val_dataset = ADE20KVal(data_root=opt.data_root, txt_file=opt.txt_file)
    elif opt.dataset == "WHUGeoGenv2":
        val_dataset = WHUGeoGenv2Val(data_root=opt.data_root, txt_file=opt.txt_file)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data in val_dataloader:
                    label = data["label"].to(device)
                    class_ids = data["class_ids"].to(device)
                    text = data["caption"]
                    c = model.get_learned_conditioning(text)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     label=label,
                                                     class_ids=class_ids,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for i in range(len(x_samples_ddim)):
                        x_sample = x_samples_ddim[i]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img_name = data["img_name"][i]
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, f"{img_name}.jpg"))

if __name__ == "__main__":
    main()
