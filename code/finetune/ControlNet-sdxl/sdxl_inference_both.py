from share import *
import config

import cv2
import einops
import torch
import random
from PIL import Image, ImageFont, ImageDraw

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldmXL.model import create_model, load_state_dict
import cv2
import json

from scripts.streamlit_helpers import *
import os

os.environ['CURL_CA_BUNDLE'] = ''


# SAVE_PATH = "outputs/demo/txt2img/"

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
}

if __name__ == "__main__":

    model = create_model('./models/cldm_xl.yaml').cpu()
    ckpt_path='./models/checkpoints_0.1_notrain_finetune_on_sdxl_bj_BJ_NY_data512_512_1e-06/epoch=4-step=112459.ckpt'
    ckpt_path_name=ckpt_path.split('/')[2]
    # print(ckpt_path_name)
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'))
    model = model.cuda()  # if you do not have enough V memory, you can comment this line load container, Unet, control_model step by step
    RESOLUTION = 512
    res = ['high', 'mid', 'low']
    # size = ['data512', 'data1024', 'data2048']
    # size = ['data1024', 'data2048']
    size = ['data512']

    data_name = ['BeiJing', 'NewYork']
    for s in size:
        for r in res:
            for d in data_name:

                path_name = f"/home/root123/mxq/data/WHUGeoGen_v2_test/test/{s}/{r}/RS_images"
                OUTPUT_DIR = f"whugeogenv2_outputs_{ckpt_path_name}_{RESOLUTION}_epoch5"
                root_name = '/home/root123/mxq/data/WHUGeoGen_v2_test'
                output_name = f"/home/root123/mxq/data/WHUGeoGen_v2_test/{OUTPUT_DIR}/sample_{RESOLUTION}/WHUGeoGen_v2_test_{s}_{r}_{d}_sample/"
                vis_name = f"/home/root123/mxq/data/WHUGeoGen_v2_test/{OUTPUT_DIR}/vis_{RESOLUTION}/WHUGeoGen_v2_test_{s}_{r}_{d}_sample/"
                if not os.path.exists(output_name):
                    os.makedirs(output_name)
                if not os.path.exists(vis_name):
                    os.makedirs(vis_name)
                with open(os.path.join(path_name, f"{d}.json"), 'rt') as f:
                    for line in f:
                        # data.append(json.loads(line))

                        data = json.loads(line)
                        prompt = data['prompt']
                        seg = data['source']
                        img = data['target']

                        # OUTPUT_DIR = f"whugeogenv2_outputs_0.2_BJ_NY_data512_256_1e-05_sdxl_bj_{RESOLUTION}_epoch1/"
                        _, file_name = os.path.split(img)
                        # os.makedirs(OUTPUT_DIR + "/" + directory_path, exist_ok=True)
                        outputfile = output_name + file_name
                        vis_file=vis_name+file_name
                        if os.path.exists(outputfile):
                            print(outputfile)
                        elif os.path.exists(vis_file):
                            print(vis_file)
                        else:
                            detect_map = cv2.imread(os.path.join(root_name, seg))
                            detect_map = cv2.cvtColor(detect_map, cv2.COLOR_BGR2RGB)
                            # 获取图像尺寸
                            height, width, channels = detect_map.shape
                            #

                            is_legacy = False
                            # W = 512
                            # H = 512
                            C = 4
                            F = 8

                            seed_everything(42)

                            value_dict = {
                                "orig_width": width,
                                "orig_height": height,
                                "target_width": RESOLUTION,
                                "target_height": RESOLUTION,
                                "crop_coords_top": 0.0,
                                "crop_coords_left": 0.0,
                                'prompt': prompt,
                                'negative_prompt': ''
                            }

                            sampler, num_rows, num_cols = init_sampling(stage2strength=None)
                            model.sampler = sampler

                            num_samples = num_rows * num_cols

                            # apply_hed = HEDdetector()

                            # detect_map = cv2.resize(detect_map, (W, H))
                            # detect_map = apply_hed(detect_map)
                            # detect_map = HWC3(detect_map)
                            # detect_map=Image.open(os.path.join(path_name,seg)).convert('RGB')
                            # detect_map_arr=np.array(detect_map)
                            # control = torch.from_numpy(cropped_image.copy()).float().cuda() / 255.0
                            if RESOLUTION == 512:
                                # control = torch.from_numpy(detect_map.copy()).float().cuda() / 255.0
                                # detect_map = cv2.resize(detect_map, (RESOLUTION, RESOLUTION), cv2.INTER_NEAREST)
                                control = torch.from_numpy(detect_map.copy()).float().cuda() / 255.0

                                control = torch.stack([control for _ in range(num_samples)], dim=0)
                                hint = einops.rearrange(control, 'b h w c -> b c h w').clone()

                                batch, batch_uc = get_batch(
                                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                                    value_dict,
                                    [num_samples],
                                )

                                force_uc_zero_embeddings = ["txt"] if not is_legacy else []

                                c, uc = model.conditioner.get_unconditional_conditioning(
                                    batch,
                                    batch_uc=batch_uc,
                                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                                )

                                # del model.conditioner

                                for k in c:
                                    if not k == "crossattn":
                                        print("ks is not crossattn")
                                        c[k], uc[k] = map(
                                            lambda y: y[k][: math.prod([num_samples])].to("cuda"), (c, uc)
                                        )

                                shape = (math.prod([num_samples]), C, RESOLUTION // F, RESOLUTION // F)

                                with torch.no_grad():
                                    with model.ema_scope():
                                        samples = model.sample(c, uc, batch_size=1, hint=hint, shape=shape)
                                        # del model.control_model
                                        # samples_copy = samples.clone()
                                        # samples_copy = einops.rearrange(samples_copy, 'b c h w -> b h w c')
                                        # samples_copy = samples_copy.cpu().numpy()
                                        # samples_copy = samples_copy * 255.0
                                        # samples_copy = samples_copy.astype(np.uint8)
                                        # cv2.imwrite("test_out.png", samples_copy[0])
                                        samples = model.decode_first_stage(samples)

                                samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

                                for sample in samples:
                                    sample = 255.0 * einops.rearrange(sample.cpu().numpy(), "c h w -> h w c")
                                    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(outputfile, sample)

                                    detect_map2 = cv2.cvtColor(detect_map, cv2.COLOR_RGB2BGR)
                                    # detect_map2_cropped_image = detect_map2[crop_rect[1]:crop_rect[1] + crop_rect[3],
                                    #                 crop_rect[0]:crop_rect[0] + crop_rect[2]]

                                    # detect_map2 = cv2.resize(detect_map2, (RESOLUTION, RESOLUTION))
                                    # samples_image = Image.fromarray(sample, "RGB")
                                    # source_seg = Image.fromarray(example["source_seg"], "RGB")
                                    # target_img = Image.fromarray(example["target_img"], "RGB")
                                    target_img = cv2.imread(os.path.join(root_name, img))
                                    # target_img_cropped_image = target_img[crop_rect[1]:crop_rect[1] + crop_rect[3],
                                    #                 crop_rect[0]:crop_rect[0] + crop_rect[2]]
                                    # target_img = cv2.resize(target_img, (RESOLUTION, RESOLUTION),cv2.INTER_NEAREST)

                                    # target_img = cv2.resize(target_img, (RESOLUTION, RESOLUTION))
                                    # target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                                    # target_img = Image.open(os.path.join(path_name, img)).convert('RGB')
                                    # target_img = np.array(target_img)
                                    # target_img = np.array(target_img)
                                    canvas = np.zeros((RESOLUTION, RESOLUTION * 3, 3), np.uint8)

                                    canvas[:RESOLUTION, :RESOLUTION] = detect_map2
                                    canvas[:RESOLUTION, RESOLUTION:2 * RESOLUTION] = target_img
                                    canvas[:RESOLUTION, 2 * RESOLUTION:3 * RESOLUTION] = sample

                                    # img2 = cv2.hconcat([detect_map2,target_img, sample])

                                    # name = (img.split('/')[-1])
                                    cv2.imwrite(vis_file, canvas)
                            else:
                                detect_map = cv2.resize(detect_map, (RESOLUTION, RESOLUTION), cv2.INTER_NEAREST)
                                control = torch.from_numpy(detect_map.copy()).float().cuda() / 255.0

                                control = torch.stack([control for _ in range(num_samples)], dim=0)
                                hint = einops.rearrange(control, 'b h w c -> b c h w').clone()

                                batch, batch_uc = get_batch(
                                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                                    value_dict,
                                    [num_samples],
                                )

                                force_uc_zero_embeddings = ["txt"] if not is_legacy else []

                                c, uc = model.conditioner.get_unconditional_conditioning(
                                    batch,
                                    batch_uc=batch_uc,
                                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                                )

                                # del model.conditioner

                                for k in c:
                                    if not k == "crossattn":
                                        print("ks is not crossattn")
                                        c[k], uc[k] = map(
                                            lambda y: y[k][: math.prod([num_samples])].to("cuda"), (c, uc)
                                        )

                                shape = (math.prod([num_samples]), C, RESOLUTION // F, RESOLUTION // F)

                                with torch.no_grad():
                                    with model.ema_scope():
                                        samples = model.sample(c, uc, batch_size=1, hint=hint, shape=shape)
                                        # del model.control_model
                                        # samples_copy = samples.clone()
                                        # samples_copy = einops.rearrange(samples_copy, 'b c h w -> b h w c')
                                        # samples_copy = samples_copy.cpu().numpy()
                                        # samples_copy = samples_copy * 255.0
                                        # samples_copy = samples_copy.astype(np.uint8)
                                        # cv2.imwrite("test_out.png", samples_copy[0])
                                        samples = model.decode_first_stage(samples)

                                samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

                                for sample in samples:
                                    sample = 255.0 * einops.rearrange(sample.cpu().numpy(), "c h w -> h w c")
                                    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(outputfile, sample)



                                    # detect_map2_cropped_image = detect_map2[crop_rect[1]:crop_rect[1] + crop_rect[3],
                                    #                 crop_rect[0]:crop_rect[0] + crop_rect[2]]
                                    detect_map2 = cv2.cvtColor(detect_map, cv2.COLOR_RGB2BGR)

                                    # detect_map2 = cv2.resize(detect_map2, (RESOLUTION, RESOLUTION))
                                    # samples_image = Image.fromarray(sample, "RGB")
                                    # source_seg = Image.fromarray(example["source_seg"], "RGB")
                                    # target_img = Image.fromarray(example["target_img"], "RGB")
                                    target_img = cv2.imread(os.path.join(root_name, img))
                                    # target_img_cropped_image = target_img[crop_rect[1]:crop_rect[1] + crop_rect[3],
                                    #                 crop_rect[0]:crop_rect[0] + crop_rect[2]]
                                    # target_img = cv2.resize(target_img, (RESOLUTION, RESOLUTION),cv2.INTER_NEAREST)

                                    target_img = cv2.resize(target_img, (RESOLUTION, RESOLUTION))
                                    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                                    # target_img = Image.open(os.path.join(path_name, img)).convert('RGB')
                                    # target_img = np.array(target_img)
                                    # target_img = np.array(target_img)
                                    canvas = np.zeros((RESOLUTION, RESOLUTION * 3, 3), np.uint8)

                                    canvas[:RESOLUTION, :RESOLUTION] = detect_map2
                                    canvas[:RESOLUTION, RESOLUTION:2 * RESOLUTION] = target_img
                                    canvas[:RESOLUTION, 2 * RESOLUTION:3 * RESOLUTION] = sample

                                    # img2 = cv2.hconcat([detect_map2,target_img, sample])


                                    # name = (img.split('/')[-1])
                                    cv2.imwrite(vis_file, canvas)
                # # merge into an image
                # joint = join(detect_map, target_img, sample, RESOLUTION, flag='horizontal')
                #
                # # add title
                # saved = Image.new("RGB", (RESOLUTION * 3, RESOLUTION + 60))
                # draw = ImageDraw.Draw(saved)
                # if RESOLUTION == 512:
                #     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24,
                #                               encoding="unic")
                # else:
                #     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12,
                #                               encoding="unic")
                # # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24, encoding="unic")
                #
                # draw.text((0, 0), f"prompt:{prompt}", (255, 255, 255), font=font)
                # draw.text((0, RESOLUTION + 30), f"Semantic Image", (255, 255, 255), font=font)
                # draw.text((RESOLUTION, RESOLUTION + 30), f"RS Image", (255, 255, 255), font=font)
                # draw.text((RESOLUTION * 2, RESOLUTION + 30), f"Generated Image", (255, 255, 255), font=font)
                #
                # bw, bh = saved.size
                # lw, lh = joint.size
                # saved.paste(joint, (bw - lw, int((bh - lh) / 2)))
                # OUTPUT_DIR = f"whugeogenv2_outputs_{RESOLUTION}/"
                # os.makedirs(OUTPUT_DIR, exist_ok=True)
                # name = (img.split('/')[-1])
                #
                # saved.save(os.path.join(OUTPUT_DIR, name))
                # print(f"Saved to: {os.path.join(OUTPUT_DIR, name)}")
