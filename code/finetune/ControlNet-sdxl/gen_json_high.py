import os
import json

def get_filenames_recursively(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


# 使用示例
res=['high','mid','low']
size=['data512','data1024','data2048']
data_name=['BeiJing','NewYork']
for r in res:
    for s in size:
        for d in data_name:
            directory = f'/home/root123/mxq/data/WHUGeoGen_v2_test/test/{s}/{r}/RS_images/{d}'  # 替换为你的目录路径
            if not os.path.exists(directory):
                print(directory, "not exists")
                continue
            filenames = get_filenames_recursively(directory)

            if r == 'high':
                count = 0
                for filename in filenames:

                    count += 1
                    if count % 50 == 0:
                        # print(f"Processed {count} files")

                        final_json={}
                        if "caption" in filename:
                            # 打开文件
                            tag_path=filename.replace("caption", "txt")
                            img_path=filename.replace("caption", "jpg")
                            seg_path=filename.replace("caption", "png")
                            seg_path=seg_path.replace("RS_images", "Semantic_masks")
                            if not os.path.exists(tag_path):
                                print(tag_path,"tag_path not exists")
                                continue
                            with open(filename, 'r') as file1:
                                # 读取第一行
                                first_caption = file1.readline()
                                caption = first_caption.strip()
                            with open(tag_path, 'r') as file2:
                                # 读取第一行
                                first_tag = file2.readline()
                                tag = first_tag.strip()
                            #mxq
                            final_json['source']=seg_path[41:]
                            final_json['target']=img_path[41:]
                            #shuang48
                            # final_json['source']=seg_path[47:]
                            # final_json['target']=img_path[47:]
                            #finetune
                            #final_json['target']=img_path[51:]

                            final_json['prompt']=caption+", "+tag
                            json_file=json.dumps(final_json)
                            with open(f'{directory}.json', 'a') as f:
                                f.write(json_file)
                                f.write("\n")
                        else:
                            print(filename,"not caption file")
            else:
                for filename in filenames:
                    final_json2 = {}
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

                        final_json2['source'] = seg_path[41:]
                        final_json2['target'] = img_path[41:]
                        # final_json['target']=img_path[51:]

                        final_json2['prompt'] = caption + ", " + tag
                        json_file = json.dumps(final_json2)
                        with open(f'{directory}.json', 'a') as f2:
                            f2.write(json_file)
                            f2.write("\n")
