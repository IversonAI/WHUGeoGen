import os
import json

def get_filenames_recursively(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


# 使用示例
res=['Airport','Port','Tank']
size=['high','mid','low']
data_name=['China','America']
for r in res:
    for s in size:
        for d in data_name:
            directory = f'/data/dailei/WHUGeoGen3/WHUGeoGenv2_focus/test/{r}/{s}/RS_images/{d}'  # 替换为你的目录路径
            if not os.path.exists(directory):
                print(directory, "not exists")
                continue
            filenames = get_filenames_recursively(directory)

            count = 0
            for filename in filenames:

                count += 1
                if count % 2 == 0:
                    # print(f"Processed {count} files")

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
                        # mxq

                        final_json['source'] = seg_path[42:]
                        final_json['target'] = img_path[42:]

                        # shuang48
                        # final_json['source']=seg_path[47:]
                        # final_json['target']=img_path[47:]
                        # finetune
                        # final_json['target']=img_path[51:]

                        final_json['prompt'] = caption + ", " + tag
                        json_file = json.dumps(final_json)
                        with open(f'{directory}.json', 'a') as f:
                            f.write(json_file)
                            f.write("\n")
                    else:
                        print(filename, "not caption file")
