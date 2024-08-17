import os
import shutil
import argparse

def move_nested_files(source_folder, destination_folder, data_name,block_name):
    # # 确保目标文件夹存在
    # os.makedirs(destination_folder, exist_ok=True)

    # block_name = 'NewYork'
    # data_name= os.path.basename(source_folder)

    block_path=os.path.join(source_folder,data_name,block_name)
    if not os.path.exists(block_path):
        print(f"block_path {block_path} not exists")
        return
    # 遍历文件夹 source_folder 中的所有文件
    for f_name in os.listdir(block_path):
        f_path = os.path.join(block_path, f_name)
        # 如果不是文件
        if not os.path.isfile(f_path):
            for name in os.listdir(f_path):
                d_path = os.path.join(f_path, name)

                if os.path.isfile(d_path):
                    # 构建目标文件路径
                    if data_name == 'BeiJing':
                        relative_path = os.path.join(f_name, data_name, block_name,name)
                        # relative_path = os.path.relpath(img_path, source_folder)
                        destination = os.path.join(destination_folder, relative_path)
                        # 移动文件
                        os.makedirs(os.path.dirname(destination), exist_ok=True)
                        shutil.copy(d_path, destination)
                        print(f"move {d_path} to {destination}")
                    if data_name == 'NewYork':
                        relative_path = os.path.join(f_name, data_name, "NY"+block_name[0:2],name)
                        # relative_path = os.path.relpath(img_path, source_folder)
                        destination = os.path.join(destination_folder, relative_path)
                        # 移动文件
                        os.makedirs(os.path.dirname(destination), exist_ok=True)
                        shutil.copy(d_path, destination)
                        print(f"move {d_path} to {destination}")



# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')
# 添加参数
parser.add_argument('--data_path', dest='folder_path', default='/data/dailei/WHUGeoGen_v2/Blip2_caption',type=str, help='input subdataset path like /home/wyd/mxq/data/OSM2/13Buffalo')
parser.add_argument('--destination_path', dest='destination_path', default='/data/dailei/WHUGeoGen3',type=str, help='input destination path like /data/dailei/WHUGeoGen3')
parser.add_argument('--data_name', dest='data_name', default='NewYork',type=str, help='input subdataset name like BeiJing')
parser.add_argument('--block_name', dest='block_name', default='11SaratogaSprings',type=str, help='input subdataset name like BJ1')

# 解析命令行参数
args = parser.parse_args()
type = ['data512', 'data1024', 'data2048']

res = ['high', 'mid', 'low']
split=['pretrain', 'finetune', 'test']
source_folder = args.folder_path # 根文件夹路径
destination_folder = args.destination_path # 目标文件夹路径
data_name = args.data_name # 数据集名
block_name = args.block_name # 块名
for s in split:
    for t in type:
        for r in res:
            base_path = os.path.join(s,t, r)
            root_path = os.path.join(source_folder, base_path)
            destination_path = os.path.join(destination_folder, base_path)
            move_nested_files(root_path, destination_path, data_name, block_name)

# find /data/dailei/WHUGeoGen3/test/data512/high/RS_images/NewYork -name *.jpg  | wc -l> /data/dailei/WHUGeoGen3/sum/test_data512_high_RS_images_NewYork.txt

