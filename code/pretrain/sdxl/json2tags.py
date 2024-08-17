from collections import defaultdict

import json
import os

import argparse

level_mapping_table = {
    '15': 'satellite image, 5m resolution, high resolution',
    '16': 'satellite image, 2m resolution, high resolution',
    '17': 'satellite image, 1m resolution, high resolution',
    '18': 'satellite image, 0.5m resolution, high resolution',
    '19': 'satellite image, 0.3m resolution, high resolution',
    '20': 'satellite image, 0.2m resolution, very high resolution',
}


def generate_tags(gen_path, out_path, ratio):
    # 遍历文件夹中的图像
    for filename in os.listdir(gen_path):
        if filename.endswith(".json") or filename.endswith(".JSON"):
            # 构建图像文件的完整路径
            json_path = os.path.join(gen_path, filename)

            # 尝试打开图片文件
            with open(json_path, "r",
                      encoding="utf-8") as json_file:  # 打开json 文件，将文件对象赋值给变量 json_file
                # 使用 json_file 变量来读取或写入文件的内容
                txt_name = filename[:-5] + '.txt'  # 将json文件名后缀改为txt

                if os.path.exists(os.path.join(out_path, txt_name)):  # 判断txt文件是否存在
                    print(f'{os.path.join(out_path, txt_name)} file exists')
                else:
                    with open(os.path.join(out_path, txt_name), 'w') as text_file:  # 打开txt文件 赋值给变量text_file
                        data = json_file.read()  # 读取文件
                        try:
                            data = json.loads(data)
                        except json.decoder.JSONDecodeError:
                            print("JSONDecodeError: No valid JSON object could be decoded from the string.")
                        # data=json.loads(data)
                        # print(data)
                        level = str(data['level'])
                        # print(level)
                        level = level_mapping_table[level]
                        tags_coarse = data['categories']
                        # print(tags_coarse)

                        tags = data['tags']
                        width = data['width']
                        height = data['height']
                        # print(width,height)
                        all_pixels = width * height
                        pixels_min = ratio * all_pixels

                        tags_raw_filtered = {}
                        for k, v in tags_coarse.items():
                            if v > 0:
                                k = k.replace('_', ' ').lower()
                                tags_raw_filtered[k] = v
                        # print(tags_coarse_filtered)
                        # tags_coarse_filtered = {k for k, v in tags_coarse.items() if v/all_pixels>0.05}
                        tags_raw_sorted = sorted(tags_raw_filtered.items(), key=lambda x: x[1], reverse=True)
                        # print(tags_coarse_sorted)
                        tags_raw = [k for k, v in tags_raw_sorted]
                        print("tags_raw", tags_raw)

                        tags_coarse_filtered = {}
                        for k, v in tags_coarse.items():
                            if v > pixels_min:
                                k = k.replace('_', ' ').lower()
                                tags_coarse_filtered[k] = v
                        # print(tags_coarse_filtered)
                        # tags_coarse_filtered = {k for k, v in tags_coarse.items() if v/all_pixels>0.05}
                        tags_coarse_sorted = sorted(tags_coarse_filtered.items(), key=lambda x: x[1], reverse=True)
                        # print(tags_coarse_sorted)
                        tags_coarse = [k for k, v in tags_coarse_sorted]
                        # print(tags_coarse)
                        print("tags", tags)

                        tags_fine_filtered = [d for d in tags if d['pixel'] > 0]
                        tags_fine_sorted = sorted(tags_fine_filtered, key=lambda x: x['pixel'], reverse=True)
                        print("tags_fine_sorted", tags_fine_sorted)
                        tags_fine_tmp = [{k: v for k, v in d.items() if k != "pixel"} for d in tags_fine_sorted]

                        tags_mid_filtered = [d for d in tags if d['pixel'] > pixels_min]
                        print("tags_mid_filtered", tags_mid_filtered)
                        tags_mid_sorted = sorted(tags_mid_filtered, key=lambda x: x['pixel'], reverse=True)
                        tags_mid_tmp = [{k: v for k, v in d.items() if k != "pixel"} for d in tags_mid_sorted]

                        print("tags_mid_tmp", tags_mid_tmp)
                        # 使用defaultdict来按照'name'键合并字典
                        tags_mid_finally = []
                        for dictionary in tags_mid_tmp:
                            for key, value in dictionary.items():
                                key = key.replace('_', '-')
                                value = value.replace('_', '-')
                                if value == 'yes':
                                    tags_mid_finally.append(key)
                                else:
                                    tags_mid_finally.append(key + ' of ' + value)
                        #
                        tags_all = []
                        for dictionary in tags_fine_tmp:
                            for key, value in dictionary.items():
                                key = key.replace('_', '-')
                                value = value.replace('_', '-')
                                if value == 'yes':
                                    tags_all.append(key)
                                else:
                                    tags_all.append(key + ' of ' + value)
                        # 将数据写入txt文件

                        text_file.write(level + ', ' + ', '.join(tags_coarse))
                        text_file.write('\n')
                        text_file.write(level + ', ' + ', '.join(tags_mid_finally))
                        text_file.write('\n')
                        text_file.write(level + ', ' + ', '.join(tags_raw))
                        text_file.write('\n')
                        text_file.write(level + ', ' + ', '.join(tags_all))


# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加参数，这里的dest是在代码中引用这个参数时使用的变量名
parser.add_argument('--data_path', dest='folder_path', default='/data/dailei/WHUGeoGen_v2/Blip2_caption',type=str, help='input subdataset path like /home/wyd/mxq/data/OSM2/13Buffalo')
parser.add_argument('--ratio', dest='ratio', default=0.001, help='input a ratio like 0.1 to control tags filtering')
parser.add_argument('--data_name', dest='data_name', default='BeiJing',type=str, help='input subdataset name like BeiJing')
parser.add_argument('--block_name', dest='block_name', default='BJ1',type=str, help='input subdataset name like BJ1')

# 解析命令行参数
args = parser.parse_args()

folder_path = args.folder_path # 根文件夹路径
data_name = args.data_name # 数据集名
block_name = args.block_name # 块名
ratio = float(args.ratio) # 过滤标签的比例
# type = ['512','1024','2048','focus']
type = ['data512', 'data1024', 'data2048']
# type = ['data1024']

# type = ['focus']
res = ['high', 'mid', 'low']

# block_name="NewYork"
# 21Olive
for t in type:
    for r in res:
        gen_path = os.path.join(folder_path, t,  r, data_name,block_name, "Annotations")
        out_path = os.path.join(folder_path, t, r, data_name,block_name, "RS_images")
        if not os.path.exists(gen_path):
            continue
        # folder_path = "/home/wyd/mxq/data/OSM2/11SaratogaSprings" + "/" + t +"/images/" + r
        generate_tags(gen_path, out_path, ratio)

# find /home/wyd/mxq/data/OSM2 -type f -name "*.zip" -exec rm -f {} \;
# find . -type f -name "*.tags" -exec sh -c 'mv "$0" "${0%.tags}.txt"' {} \;