from collections import defaultdict

import json
import os
import shutil
import argparse

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加参数，这里的dest是在代码中引用这个参数时使用的变量名
parser.add_argument('--data_path', dest='folder_path', default='/data/dailei/WHUGeoGen_v2/Blip2_caption',type=str, help='input subdataset path like /home/wyd/mxq/data/OSM2/13Buffalo')
# parser.add_argument('--ratio', dest='ratio', default=0.001, help='input a ratio like 0.1 to control tags filtering')

# 解析命令行参数
args = parser.parse_args()

folder_path = args.folder_path # 根文件夹路径
# ratio = args.ratio
# type = ['512','1024','2048','focus']
type = ['data512', 'data1024', 'data2048']
# type = ['data1024']

# type = ['focus']
res = ['high', 'mid', 'low']

block_name="BeiJing"

for t in type:
    for r in res:
        # gen_path = os.path.join(folder_path, t,  r, block_name, "Annotations")
        gen_path = os.path.join(folder_path, t, r, block_name)
        if not os.path.exists(gen_path):
            continue
        # folder_path = "/home/wyd/mxq/data/OSM2/11SaratogaSprings" + "/" + t +"/images/" + r
        # 遍历文件夹中的图像
        for filename in os.listdir(gen_path):
            if filename=="BJ2"or filename=="BJ5"or filename=="BJ6"or filename=="BJ7"or filename=="BJ8"or filename=="BJ9"or filename=="BJ10" or filename=="BJ11"or filename=="BJ12"or filename=="BJ13"or filename=="BJ14"or filename=="BJ15"or filename=="BJ16"or filename=="BJ17"or filename=="BJ18"or filename=="BJ19"or filename=="BJ20"or filename=="BJ21":
                file_path = os.path.join(gen_path, filename)
                print(f'Deleting {file_path}')
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

# find /home/wyd/mxq/data/OSM2 -type f -name "*.zip" -exec rm -f {} \;