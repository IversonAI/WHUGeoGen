import os
import shutil
import json

def copy_images(source_folder, destination_folder, file_paths):
    for file_path in file_paths:
        # 获取文件名
        filename = os.path.basename(file_path)
        # tmp_path = os.path.dirname(file_path)
        # tmp_path = os.path.join(destination_folder, tmp_path)
        # if not os.path.exists(tmp_path):
        #     os.makedirs(tmp_path)
        # 目标文件路径
        target_path = os.path.join(destination_folder, filename)
        # 复制文件
        shutil.copy2(os.path.join(source_folder, file_path), target_path)
        print(f"Copied {file_path} to {destination_folder}")


# 使用示例
# root_path="/home/root123/mxq/data/WHUGeoGen_v2_test"
res=['high','mid','low']
size=['Tank']
data_name=['China','America']
for s in size:
    for r in res:
        for d in data_name:
            source_folder = '/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus'  # 源文件夹路径
            destination_folder = f'/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus/test/WHUGeoGen_v2_test_{s}_{r}_{d}_focus_eval'  # 目标文件夹路径
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            # file_paths = ['path/to/source/folder/image1.jpg', 'path/to/source/folder/image2.png']  # 图片文件路径列表
            file_paths = []
            path_name = f"/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus/test/{s}/{r}/RS_images"
            with open(os.path.join(path_name, f"{d}.json"), 'rt') as f:
                for line in f:
                    # data.append(json.loads(line))
                    data = json.loads(line)
                    # prompt = data['prompt']
                    # seg = data['source']
                    img = data['target'].lstrip('/')
                    file_paths.append(img)
                    # p=os.path.join(root_path, img)
                    # file_paths.append(os.path.join(root_path, img))
            copy_images(source_folder, destination_folder, file_paths)