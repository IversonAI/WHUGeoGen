import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/sior/sior_controlnet_seg.json', 'rt') as f:
        # with open('./training/loveDA/rgb_fine/train/blip2_caption_controlnet_fine_seg_loveDA_train.json', 'rt') as f:
        # with open('./training/loveDA/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_loveDA_train.json', 'rt') as f:

        # with open('./training/MiniFrance/rgb_fine/train/blip2_caption_controlnet_fine_seg_MiniFrance_train.json', 'rt') as f:
        # with open('./training/MiniFrance/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_MiniFrance_train.json', 'rt') as f:
        # with open('./training/MiniFrance/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_MiniFrance_train.json', 'rt') as f:

        # with open('./training/iSAID/rgb_fine/train/blip2_caption_controlnet_fine_seg_iSAID_train.json', 'rt') as f:
        # with open('./training/iSAID/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_iSAID_train.json', 'rt') as f:
        # with open('./training/iSAID/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_iSAID_train.json', 'rt') as f:

        # with open('./training/HRSCD/rgb_fine/train/blip2_caption_controlnet_fine_seg_HRSCD_train.json', 'rt') as f:
        # with open('./training/HRSCD/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_HRSCD_train.json', 'rt') as f:
        # with open('./training/HRSCD/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_HRSCD_train.json', 'rt') as f:

        # with open('./training/HRSCD/rgb_fine/train/blip2_caption_controlnet_fine_seg_HRSCD_train.json', 'rt') as f:
        # with open('./training/HRSCD/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_HRSCD_train.json', 'rt') as f:
        # with open('./training/HRSCD/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_HRSCD_train.json', 'rt') as f:

        # with open('./training/DroneDeploy/rgb_fine/train/blip2_caption_controlnet_fine_seg_DroneDeploy_train.json', 'rt') as f:
        # with open('./training/DroneDeploy/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_DroneDeploy_train.json', 'rt') as f:
        # with open('./training/DroneDeploy/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_DroneDeploy_train.json', 'rt') as f:

        # with open('./training/enviroatlas_landcover_nlcd/rgb_fine/train/blip2_caption_controlnet_fine_seg_enviroatlas_landcover_nlcd_train.json', 'rt') as f:
        # with open('./training/enviroatlas_landcover_nlcd/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_enviroatlas_landcover_nlcd_train.json', 'rt') as f:
        # with open('./training/enviroatlas_landcover_nlcd/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_enviroatlas_landcover_nlcd_train.json', 'rt') as f:

        # with open('./training/cvpr_chesapeake_landcover_nlcd/rgb_fine/train/blip2_caption_controlnet_fine_seg_chesapeake_train.json', 'rt') as f:
        # with open('./training/cvpr_chesapeake_landcover_nlcd/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_chesapeake_train.json', 'rt') as f:
        # with open('./training/cvpr_chesapeake_landcover_nlcd/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_chesapeake_train.json', 'rt') as f:

        #NYS16
        # with open('./training/NYS16/rgb_fine/train/blip2_caption_controlnet_fine_seg_NYS16_train.json', 'rt') as f:
        # with open('./training/NYS16/rgb_landuse/train/blip2_caption_controlnet_landuse_seg_NYS16_train.json', 'rt') as f:
        with open('./training/NYS16/rgb_coarse/train/blip2_caption_controlnet_coarse_seg_NYS16_train.json', 'rt') as f:


        ##open
        # with open('./training/rgb_fine_train.json', 'rt') as f:
        # with open('./training/rgb_landuse_train.json', 'rt') as f:
        # with open('./training/rgb_coarse_train.json', 'rt') as f:

        ###open with nys16
        # with open('./training/rgb_fine_train_with_nys16.json', 'rt') as f:
        # with open('./training/rgb_landuse_train_with_nys16.json', 'rt') as f:
        # with open('./training/rgb_coarse_train_with_nys16.json', 'rt') as f:

            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('./training/sior/' + source_filename)
        # target = cv2.imread('./training/sior/' + target_filename)

        # #  /loveDA/rgb_fine/train
        # source = cv2.imread('./training/loveDA/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/loveDA/rgb_fine/train/' + target_filename)
        # #  /loveDA/rgb_landuse/train
        # source = cv2.imread('./training/loveDA/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/loveDA/rgb_landuse/train/' + target_filename)
        # #  /loveDA/rgb_coarse/train
        # source = cv2.imread('./training/loveDA/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/loveDA/rgb_coarse/train/' + target_filename)

        # # #  /training/MiniFrance/gray_fine/train
        # source = cv2.imread('./training/MiniFrance/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/MiniFrance/rgb_fine/train/' + target_filename)

        # #  /training/MiniFrance/gray_landuse/train
        # source = cv2.imread('./training/MiniFrance/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/MiniFrance/rgb_landuse/train/' + target_filename)
        ##  /training/MiniFrance/rgb_coarse/train
        # source = cv2.imread('./training/MiniFrance/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/MiniFrance/rgb_coarse/train/' + target_filename)

        # # #  /training/iSAID/gray_fine/train
        # source = cv2.imread('./training/iSAID/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/iSAID/rgb_fine/train/' + target_filename)
        # #  /training/iSAID/gray_landuse/train
        # source = cv2.imread('./training/iSAID/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/iSAID/rgb_landuse/train/' + target_filename)
        ##  /training/iSAID/rgb_coarse/train
        # source = cv2.imread('./training/iSAID/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/iSAID/rgb_coarse/train/' + target_filename)

        # # #  /training/Geonrw/gray_fine/train
        # source = cv2.imread('./training/Geonrw/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/Geonrw/rgb_fine/train/' + target_filename)
        #  /training/Geonrw/gray_landuse/train
        # source = cv2.imread('./training/Geonrw/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/Geonrw/rgb_landuse/train/' + target_filename)
        #  /training/Geonrw/rgb_coarse/train
        # source = cv2.imread('./training/Geonrw/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/Geonrw/rgb_coarse/train/' + target_filename)

        # # #  /training/HRSCD/gray_fine/train
        # source = cv2.imread('./training/HRSCD/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/HRSCD/rgb_fine/train/' + target_filename)
        #  /training/HRSCD/gray_landuse/train
        # source = cv2.imread('./training/HRSCD/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/HRSCD/rgb_landuse/train/' + target_filename)
        #  /training/HRSCD/rgb_coarse/train
        # source = cv2.imread('./training/HRSCD/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/HRSCD/rgb_coarse/train/' + target_filename)

        # # #  /training/DroneDeploy/gray_fine/train
        # source = cv2.imread('./training/DroneDeploy/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/DroneDeploy/rgb_fine/train/' + target_filename)
        #  /training/DroneDeploy/gray_landuse/train
        # source = cv2.imread('./training/DroneDeploy/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/DroneDeploy/rgb_landuse/train/' + target_filename)
        #  /training/DroneDeploy/rgb_coarse/train
        # source = cv2.imread('./trainingraining/DroneDeploy/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/DroneDeploy/rgb_coarse/train/' + target_filename)

        # # #  /training/enviroatlas_landcover_nlcd/gray_fine/train
        # source = cv2.imread('./training/enviroatlas_landcover_nlcd/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/enviroatlas_landcover_nlcd/rgb_fine/train/' + target_filename)
        ##  /training/enviroatlas_landcover_nlcd/gray_landuse/train
        # source = cv2.imread('./training/enviroatlas_landcover_nlcd/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/enviroatlas_landcover_nlcd/rgb_landuse/train/' + target_filename)
        #  /training/enviroatlas_landcover_nlcd/rgb_coarse/train
        # source = cv2.imread('./training/enviroatlas_landcover_nlcd/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/enviroatlas_landcover_nlcd/rgb_coarse/train/' + target_filename)

        # # #  /training/cvpr_chesapeake_landcover_nlcd/gray_fine/train
        # source = cv2.imread('./training/cvpr_chesapeake_landcover_nlcd/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/cvpr_chesapeake_landcover_nlcd/rgb_fine/train/' + target_filename)
        #  /training/cvpr_chesapeake_landcover_nlcd/gray_landuse/train
        # source = cv2.imread('./training/cvpr_chesapeake_landcover_nlcd/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/cvpr_chesapeake_landcover_nlcd/rgb_landuse/train/' + target_filename)
        #  /training/cvpr_chesapeake_landcover_nlcd/rgb_coarse/train
        # source = cv2.imread('./training/cvpr_chesapeake_landcover_nlcd/rgb_coarse/train/' + source_filename)
        # target = cv2.imread('./training/cvpr_chesapeake_landcover_nlcd/rgb_coarse/train/' + target_filename)


        # # #  /training/NYS16/gray_fine/train
        # source = cv2.imread('./training/NYS16/rgb_fine/train/' + source_filename)
        # target = cv2.imread('./training/NYS16/rgb_fine/train/' + target_filename)
        #  /training/NYS16/gray_landuse/train
        # source = cv2.imread('./training/NYS16/rgb_landuse/train/' + source_filename)
        # target = cv2.imread('./training/NYS16/rgb_landuse/train/' + target_filename)
        #  /training/NYS16/rgb_coarse/train
        source = cv2.imread('./training/NYS16/rgb_coarse/train/' + source_filename)
        target = cv2.imread('./training/NYS16/rgb_coarse/train/' + target_filename)

        # # #  /training/gray_fine/train
        # source = cv2.imread('./training/' + source_filename)
        # target = cv2.imread('./training/' + target_filename)
        #  /training/gray_landuse/train
        # source = cv2.imread('./training/' + source_filename)
        # target = cv2.imread('./training/' + target_filename)
        #  /training/rgb_coarse/train
        # source = cv2.imread('./training/' + source_filename)
        # target = cv2.imread('./training/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # source = cv2.resize(source, (512, 512))
        # target = cv2.resize(target, (512, 512))

        source = cv2.resize(source, (256, 256),cv2.INTER_NEAREST) #3-35changed to cv2.INTER_NEAREST
        target = cv2.resize(target, (256, 256),cv2.INTER_NEAREST)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

