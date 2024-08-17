from share import *
import config
import cv2
import einops
import numpy as np
import torch
import random
import os
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
# from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

print("create_model cldm_v15\n")
model = create_model("../models/cldm_v15.yaml").cpu()
# print("load state dict control_sd15_seg.pth\n")
model.load_state_dict(load_state_dict("../models/control_NYS16_fine_256.pth", location='cuda'))
print("convert model to cuda\n")
model = model.cuda()
print("create DDIMSampler from model\n")
ddim_sampler = DDIMSampler(model)


def process(detected_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps,
            guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(detected_map, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


img = cv2.imread("/home/user/dailei/Code_4090/ControlNet/training/NYS16/rgb_fine/val/source/seg/16_23594_19395.png")
prompt = "an aerial view of a highway in the middle of a green field"
a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples = 1
image_resolution = 256
detect_resolution = 1024
ddim_steps = 20
guess_mode = False
strength = 1
scale = 9
seed = 106268514
eta = 0.0
result = process(img, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps,
                 guess_mode, strength, scale, seed, eta)
cv2.imwrite("./results/16_23594_19395.jpg", result[0])

'''
obj = Seg2Image()
with open("e:/CtlNet/seg.png", 'rb') as f:
    datas = f.read()
pic = obj.run(datas,"a satellite image shows a large area of trees and buildings")
print(type(pic))
with open("e:/CtlNet/out.jpg","wb") as f:
    f.write(pic)
    

img = cv2.imread("f:/controlnet/seg.png") 
prompt="building"
a_prompt="best quality, extremely detailed"
n_prompt ="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples= 1
image_resolution=512
detect_resolution=256
ddim_steps=20
guess_mode=False
strength=1
scale=9
seed=106268514
eta=0.0
result = process(img,prompt,a_prompt,n_prompt,num_samples,image_resolution,detect_resolution,ddim_steps,guess_mode,strength,scale,seed,eta)
cv2.imwrite("f:/controlnet/autoresult.png", result[0])
'''
