from transformers import CLIPProcessor, CLIPModel
import torch
import os
import cv2

import json

# # 加载CLIP模型
# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
#
#
# def calculate_clip_score(images, prompts):
#     # import pdb;pdb.set_trace()
#     # images_int = (np.asarray(images[0]) * 255).astype("uint8")
#     images_int = (np.asarray(images)).astype("uint8")
#     clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
#     return round(float(clip_score), 4)
#
#
# # 设定随机数种子，规定随机数。  从而使每次执行同样的prompt，生成图片一样
# torch.manual_seed(0)
#
# # load a StableDiffusionPipeline
# # 指定加载的stable diffusion模型名称
# model_id = "stabilityai/stable-diffusion-2-1-base"
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda:0")
# Load the CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
model.to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# RESOLUTION = 512
res = ['high', 'mid', 'low']
# size = ['data512', 'data1024', 'data2048']
# size = ['data1024', 'data2048']
size = ['data512']

data_name = ['BeiJing', 'NewYork']
for s in size:
    for r in res:
        for d in data_name:

            path_name = f"/home/root123/mxq/data/WHUGeoGen_v2_test/test/{s}/{r}/RS_images"
            # OUTPUT_DIR = f"whugeogenv2_outputs_{ckpt_path_name}_{RESOLUTION}_epoch5"
            root_name = '/home/root123/mxq/data/WHUGeoGen_v2_test/whugeogenv2_outputs_checkpoints_0.1_notrain_finetune_on_sdxl_bj_BJ_NY_data512_256_1e-06_512_epoch5/sample_512'
            output_name = f"WHUGeoGen_v2_test_{s}_{r}_{d}_sample/"
            output_name_2 = output_name.replace("/", "")
            # vis_name = f"/home/root123/mxq/data/WHUGeoGen_v2_test/{OUTPUT_DIR}/vis_{RESOLUTION}/WHUGeoGen_v2_test_{s}_{r}_{d}_sample/"
            # if not os.path.exists(output_name):
            #     os.makedirs(output_name)
            # if not os.path.exists(vis_name):
            #     os.makedirs(vis_name)
            clips = 0.0
            count = 0
            sum = []
            with open(os.path.join(path_name, f"{d}.json"), 'rt') as f:
                for line in f:
                    # data.append(json.loads(line))

                    data = json.loads(line)
                    prompt = data['prompt']
                    # seg = data['source']
                    img = data['target']

                    # OUTPUT_DIR = f"whugeogenv2_outputs_0.2_BJ_NY_data512_256_1e-05_sdxl_bj_{RESOLUTION}_epoch1/"
                    _, file_name = os.path.split(img)
                    # os.makedirs(OUTPUT_DIR + "/" + directory_path, exist_ok=True)
                    outputfile = output_name + file_name
                    img_path = os.path.join(root_name, outputfile)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.transpose((2, 0, 1))
                    # print(image.shape)
                    image = torch.from_numpy(image[None, :, :])

                    # print(image.shape)
                    # CLIP分数计算
                    with torch.no_grad():
                        inputs = processor(prompt, images=image, truncation=True, max_length=77, return_tensors="pt",
                                           padding=True).to(device)
                        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
                        text_features = model.get_text_features(input_ids=inputs.input_ids,
                                                                attention_mask=inputs.attention_mask)

                        # Calculate the cosine similarity
                        cosine_similarity = torch.nn.functional.cosine_similarity(image_features,
                                                                                  text_features).item()
                    # sd_clip_score = calculate_clip_score(image, prompt)
                    print(cosine_similarity)
                    count += 1
                    clips = clips + cosine_similarity
                    sum.append([img_path, cosine_similarity])
            mean_clips = clips / count
            print(f"mean CLIP score: {mean_clips}")
            with open(os.path.join(root_name, f"{output_name_2}_clips.txt"), 'w') as f2:
                f2.write(str(mean_clips))

            with open(os.path.join(root_name, f"{output_name_2}_sum.txt"), 'w') as f3:
                for i in sorted(sum):
                    f3.write(str(i) + '\n')
# prompt = "wood"
# images = pipe(prompt, num_images_per_prompt=1, output_type="numpy").images
# # 保存图片，此步骤对于计算clip score不是必要的。
# images[0].save("wood.png")
