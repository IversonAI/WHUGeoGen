# import os
# import glob
#
#
# def get_file_paths(directory, folder_name):
#     """
#     获取directory中所有folder_name文件夹中文件的路径，并将它们存储到一个列表中。
#     """
#     jpg_paths = []
#
#     for root, dirs, files in os.walk(directory):
#         if folder_name in dirs:
#             folder_path = os.path.join(root, folder_name)
#             for file in files:
#                 if file.endswith('.jpg'):
#                     jpg_paths.append(os.path.join(folder_path, file))
#     return jpg_paths
#
#
# # 使用示例
# directory = '/home/root123/Dailei/01_Data/WHUGeoGen_v2_finetune/finetune/data512'  # 嵌套文件夹的路径
# folder_name = 'RS_images'  # 需要搜索的特定文件夹名称
# file_paths = get_file_paths(directory, folder_name)
# print(file_paths)
#
# import os
#
# # 定义图片文件夹路径
# folder_path = 'RS_images'
#
# # 初始化一个空列表来存储图片路径
# jpg_paths = []
#
# # 确保文件夹路径是正确的
# if os.path.isdir(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.jpg'):
#             jpg_paths.append(os.path.join(folder_path, filename))
#
# # 打印jpg_paths列表
# print(jpg_paths)

import os
import json

def get_filenames_recursively(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


# 使用示例
size=['Tank']
for s in size:
    directory = f'/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus/test/{s}'  # 替换为你的目录路径
    filenames = get_filenames_recursively(directory)
    for filename in filenames:

        final_json={}
        if "caption" in filename:
            # 打开文件
            tag_path=filename.replace("caption", "txt")
            img_path=filename.replace("caption", "jpg")
            seg_path=filename.replace("caption", "png")
            seg_path=seg_path.replace("RS_images", "Semantic_masks")
            with open(filename, 'r') as file1:
                # 读取第一行
                first_caption = file1.readline()
                caption = first_caption.strip()
            if not os.path.exists(tag_path):
                print(tag_path,"tag_path not exists")
                continue
            with open(tag_path, 'r') as file2:
                # 读取第一行
                first_tag = file2.readline()
                tag = first_tag.strip()

            final_json['source']=seg_path[42:]
            final_json['target']=img_path[42:]
            final_json['prompt']=caption+", "+tag
            json_file1=json.dumps(final_json)
            with open(f'{directory}.json', 'a') as f1:
                f1.write(json_file1)
                f1.write("\n")
            final_json['prompt']=caption
            json_file2=json.dumps(final_json)
            with open(f'{directory}_caption.json', 'a') as f2:
                f2.write(json_file2)
                f2.write("\n")
            final_json['prompt']=tag
            json_file3=json.dumps(final_json)
            with open(f'{directory}_tag.json', 'a') as f3:
                f3.write(json_file3)
                f3.write("\n")
