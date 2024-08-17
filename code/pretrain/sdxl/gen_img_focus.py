from diffusers import StableDiffusionXLPipeline
import torch
import gc
import time as time_
import random
import os
import json

def inference(seed=-1,prompt=None,img=None,r=512):
    access_token = ''

    # model_key_base = "/home/user/dailei/Code_4090/sd-scripts-0.8.7/outputs/sdxl_bj.safetensors"
    # model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"
    # import torch
    # from diffusers import StableDiffusionXLPipeline
    # pretrained_ckpt_path = "/data/xiedong/fooocus_tensorRT/Fooocus/models/checkpoints/juggernautXL_version6Rundiffusion.safetensors"


    if seed == -1:
        seed = int(random.randrange(4294967294))

    device = 'cuda'
    generator = torch.Generator(device=device)

    generator = generator.manual_seed(seed)

    latents = torch.randn(
        (1, pipe.unet.in_channels, r // 8, r // 8),
        generator=generator,
        device=device,
        dtype=torch.float16
    )


    # prompt = '✨aesthetic✨ aliens walk among us in Las Vegas, scratchy found film photograph'
    prompt = prompt
    negative_prompt = 'low quality'
    guidance_scale = 7
    num_inference_steps = 20

    images = pipe(prompt=[prompt], negative_prompt=[negative_prompt],
                  guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                  latents=latents).images

    gc.collect()
    torch.cuda.empty_cache()

    images[0].save(img)


if __name__ == "__main__":
    start_time = time_.time()
    model_key_base = "/home/user/dailei/Code_4090/sd-scripts-0.8.7/outputs/sdxl_bj_ny.safetensors"
    pipe = StableDiffusionXLPipeline.from_single_file(model_key_base, torch_dtype=torch.float16).to("cuda")
    print("Loading model", model_key_base)

    # pipe = DiffusionPipeline.from_pretrained(
    #     model_key_base, torch_dtype=torch.float16, resume_download=True, use_auth_token=access_token)
    # pipe.enable_model_cpu_offload()
    ckpt_path_name=model_key_base.split("/")[-1].split(".")[0]
    RESOLUTION = 512
    size = ['Airport', 'Port', 'Tank']
    # size = ['data512', 'data1024', 'data2048']
    # size = ['data1024', 'data2048']
    res = ['high', 'low', 'mid']

    data_name = ['China', 'America']
    for s in size:
        for r in res:
            for d in data_name:
                path_name = f"/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus/test/{s}/{r}/RS_images"
                OUTPUT_DIR = f"whugeogenv2_t2i_{ckpt_path_name}_{RESOLUTION}_epoch5"
                root_name = '/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus'
                output_name = f"/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus/{OUTPUT_DIR}/sample_{RESOLUTION}/WHUGeoGen_v2_focus_test_{s}_{r}_{d}_sample/"
                # vis_name = f"/home/root123/mxq/data/WHUGeoGen_v2_test/{OUTPUT_DIR}/vis_{RESOLUTION}/WHUGeoGen_v2_test_{s}_{r}_{d}_sample/"
                if not os.path.exists(output_name):
                    os.makedirs(output_name)
                # if not os.path.exists(vis_name):
                #     os.makedirs(vis_name)
                with open(os.path.join(path_name, f"{d}.json"), 'rt') as f:
                    for line in f:
                        # data.append(json.loads(line))

                        data = json.loads(line)
                        prompt = data['prompt']
                        # seg = data['source']
                        img2 = data['target']

                        # OUTPUT_DIR = f"whugeogenv2_outputs_0.2_BJ_NY_data512_256_1e-05_sdxl_bj_{RESOLUTION}_epoch1/"
                        _, file_name = os.path.split(img2)
                        # os.makedirs(OUTPUT_DIR + "/" + directory_path, exist_ok=True)
                        outputfile = output_name + file_name
                        # vis_file = vis_name + file_name
                        if os.path.exists(outputfile):
                            print(outputfile)
                        # elif os.path.exists(vis_file):
                        #     print(vis_file)
                        inference(-1,  prompt=prompt, img=outputfile,r=RESOLUTION)
    # Run your code
    # inference(-1)

    end_time = time_.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")