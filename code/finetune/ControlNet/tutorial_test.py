from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained("./models/control_sd15_ini.ckpt", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "./checkpoints/epoch=20-step=19634.ckpt", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

control_image = load_image("/data/dailei/WHUGeoGen/MiniFrance/rgb_fine/val/source/seg/10009.png")
prompt = "an aerial view of a swampy area with lots of trees"

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]
image.save("./output.png")