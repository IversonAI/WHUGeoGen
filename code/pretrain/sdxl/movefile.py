import os
import shutil
import argparse

def move_nested_files(source_folder, destination_folder, block_name):
    # # 确保目标文件夹存在
    # os.makedirs(destination_folder, exist_ok=True)

    # block_name = 'NewYork'
    data_name= os.path.basename(source_folder)
    # 遍历文件夹 source_folder 中的所有文件
    for size_name in os.listdir(source_folder):
        size_path = os.path.join(source_folder, size_name)
        # 如果不是文件
        if not os.path.isfile(size_path):
            # 遍历文件夹 size_path 中的所有文件
            for file_name in os.listdir(size_path):
                file_path = os.path.join(size_path, file_name)
                # 如果不是文件
                if not os.path.isfile(file_path):
                    # 遍历文件夹 file_path 中的所有文件
                    for res_name in os.listdir(file_path):
                        res_path = os.path.join(file_path, res_name)
                        # 如果不是文件
                        if not os.path.isfile(res_path):
                            # 遍历文件夹 res_path 中的所有文件
                            for img_name in os.listdir(res_path):
                                img_path = os.path.join(res_path, img_name)
                                # 如果是文件
                                if os.path.isfile(img_path):
                                    # 构建目标文件路径
                                    if file_name == 'images':
                                        file_name = 'RS_images'
                                    relative_path= os.path.join(size_name,res_name,block_name,data_name,file_name,img_name)
                                    # relative_path = os.path.relpath(img_path, source_folder)
                                    destination = os.path.join(destination_folder, relative_path)
                                    # 移动文件
                                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                                    shutil.move(img_path, destination)

# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')
# 添加参数
parser.add_argument('--data_path', dest='folder_path', default='/data/dailei/WHUGeoGen/NY22/11SaratogaSprings',type=str, help='input subdataset path like /home/wyd/mxq/data/OSM2/13Buffalo')
parser.add_argument('--destination_path', dest='destination_path', default='/data/dailei/WHUGeoGen3',type=str, help='input destination path like /data/dailei/WHUGeoGen3')
parser.add_argument('--block_name', dest='block_name', default='NewYork',type=str, help='input subdataset name like NewYork')

# 解析命令行参数
args = parser.parse_args()

source_folder = args.folder_path # 根文件夹路径
destination_folder = args.destination_path # 目标文件夹路径
block_name = args.block_name # 块名
move_nested_files(source_folder, destination_folder, block_name)