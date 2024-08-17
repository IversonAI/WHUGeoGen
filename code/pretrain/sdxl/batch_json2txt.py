import os
import re
import json
from collections import defaultdict

level_mapping_table = {
    '15': 'satellite image, 5m resolution, high resolution',
    '16': 'satellite image, 2m resolution, high resolution',
    '17': 'satellite image, 1m resolution, high resolution',
    '18': 'satellite image, 0.5m resolution, high resolution',
    '19': 'satellite image, 0.3m resolution, high resolution',
    '20': 'satellite image, 0.2m resolution, very high resolution',
}

# 读取文件夹中的文件名
path='/media/wyd-7/ba7c6ce2-33ae-42f2-9777-a6aa32ab2e66/TEST_NY/output_cu_2048/Annotations_2048'
fileList = os.listdir(path)
print(fileList)

# 循环读取文件，并请求
for json_name in fileList:
    if not json_name.endswith('.json'):  # 判断文件是否为json文件
        continue
    print(json_name)
    with open(os.path.join(path,json_name),"r", encoding="utf-8") as json_file:  # 打开json 文件，将文件对象赋值给变量 json_file
        # 使用 json_file 变量来读取或写入文件的内容
        txt_name=json_name[:-5]+'.txt'  # 将json文件名后缀改为txt

        if os.path.exists(os.path.join(path,txt_name)):  # 判断txt文件是否存在
            print(f'{txt_name} file exists')
        else:
            with open(os.path.join(path,txt_name), 'w') as text_file:  # 打开txt文件 赋值给变量text_file
                data = json_file.read()  # 读取文件
                try:
                    parsed_data = json.loads(data)
                except json.decoder.JSONDecodeError:
                    print("JSONDecodeError: No valid JSON object could be decoded from the string.")
                # data=json.loads(data)
                # print(type(data))
                level=str(data['level'])
                # print(level)
                level=level_mapping_table[level]
                tags_coarse=data['categories']
                # print(tags_coarse)

                tags2=data['tags']
                # print(tags2)
                # 使用defaultdict来按照'name'键合并字典
                merged_dict = defaultdict(list)

                for d in tags2:
                    for k, v in d.items():
                        if v != 'yes':
                            merged_dict[k].append(v)

                # 将defaultdict转换为普通字典
                merged_dict = {k: list(set(v)) for k, v in merged_dict.items()}

                tags_fine = []

                for key, value in merged_dict.items():
                    # print(key, value)
                    key=key.replace('_','-')
                    value = [v.replace('_', '-') for v in value]
                    if len(value) == 1:
                        tags_fine.append(key + ' of ' + value[0])
                    else:
                        tags_fine.append(key + ' of {' + '|'.join(value)+'}')

                tags_all = []
                for dictionary in tags2:
                    for key, value in dictionary.items():
                        key = key.replace('_', '-')
                        value = value.replace('_', '-')
                        if value!= 'yes':
                            tags_all.append(key + ' of ' + value)
                # 将数据写入txt文件
                text_file.write(level + ', ' + ', '.join(tags_coarse))
                text_file.write('\n')
                text_file.write(level + ', ' + ', '.join(tags_fine))
                text_file.write('\n')
                text_file.write(level + ', ' + ', '.join(tags_all))
