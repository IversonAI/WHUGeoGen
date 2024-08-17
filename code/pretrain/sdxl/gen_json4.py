import os
import json


def get_filenames_recursively(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


# 使用示例
# res=['high','mid','low']
res = ['data512', 'data768', 'data1024', 'data1280', 'data2048']
split = ['finetune', 'test']

for r in res:
    for s in split:
        # 使用示例
        directory = f'/home/root123/00Dailei/01Data/WHUGeoGenv2/{s}/{r}'  # 替换为你的目录路径
        filenames = get_filenames_recursively(directory)
        for filename in filenames:

            final_json = {}
            if "caption" in filename:
                # 打开文件
                tag_path = filename.replace("caption", "txt")
                img_path = filename.replace("caption", "jpg")
                seg_path = filename.replace("caption", "png")
                seg_path = seg_path.replace("RS_images", "Semantic_masks")
                if not os.path.exists(tag_path):
                    print(tag_path, "tag_path not exists")
                    continue
                with open(filename, 'r') as file1:
                    # 读取第一行
                    first_caption = file1.readline()
                    caption = first_caption.strip()
                with open(tag_path, 'r') as file2:
                    # 读取第一行
                    first_tag = file2.readline()
                    tag = first_tag.strip()

                if s == 'test':
                    final_json['source'] = seg_path[47:]
                    final_json['target'] = img_path[47:]
                else:
                    final_json['source'] = seg_path[51:]
                    final_json['target'] = img_path[51:]
                # final_json['source']=seg_path[47:]
                # final_json['target']=img_path[47:]
                # final_json['target']=img_path[51:]

                final_json['prompt']=caption+", "+tag
                json_file=json.dumps(final_json)
                with open(f'{directory}_BJ_NY.json', 'a') as f:
                    f.write(json_file)
                    f.write("\n")

                # final_json['prompt'] = caption
                # json_file = json.dumps(final_json)
                # with open(f'{directory}_BJ_NY_caption.json', 'a') as f:
                #     f.write(json_file)
                #     f.write("\n")
                #
                # final_json['prompt'] = tag
                # json_file2 = json.dumps(final_json)
                # with open(f'{directory}_BJ_NY_tag.json', 'a') as f2:
                #     f2.write(json_file2)
                #     f2.write("\n")
