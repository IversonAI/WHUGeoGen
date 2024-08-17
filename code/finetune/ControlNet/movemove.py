import os
import shutil
import re

# 定义图像文件夹路径和图像名称的正则表达式
image_folder = '/home/user/dailei/Code_4090/FreestyleNet/outputs/WHUGeoGenv2_LIS_freestyle/WHUGeoGenv2_LIS_512_t2i'
output_folder = '/home/user/dailei/Code_4090/FreestyleNet/outputs_eval/WHUGeoGenv2_LIS_freestyle/WHUGeoGenv2_LIS_512_t2i'
# name_pattern = re.compile(r'pattern_to_extract_from_name')

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历图像文件夹中的所有图像文件
for filename in os.listdir(image_folder):
    # 获取图像名称的匹配结果
    # match = name_pattern.search(filename)
    if "high" in filename and "BeiJing" in filename:
        # 根据匹配结果创建文件夹并移动图像
        folder_name = "high_BeiJing"
        out_filename = filename.split("_")[-3:]
        out_filename = "_".join(out_filename)
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(output_subfolder, out_filename)
        shutil.copyfile(source_path, destination_path)
    elif "mid" in filename and "BeiJing" in filename:
        folder_name = "mid_BeiJing"
        out_filename = filename.split("_")[-3:]
        out_filename = "_".join(out_filename)
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(output_subfolder, out_filename)
        shutil.copyfile(source_path, destination_path)
    elif "low" in filename and "BeiJing" in filename:
        folder_name = "low_BeiJing"
        out_filename = filename.split("_")[-3:]
        out_filename = "_".join(out_filename)
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(output_subfolder, out_filename)
        shutil.copyfile(source_path, destination_path)
    elif "high" in filename and "NewYork" in filename:
        folder_name = "high_NewYork"
        out_filename = filename.split("_")[-3:]
        out_filename = "_".join(out_filename)
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(output_subfolder, out_filename)
        shutil.copyfile(source_path, destination_path)
    elif "mid" in filename and "NewYork" in filename:
        folder_name = "mid_NewYork"
        out_filename = filename.split("_")[-3:]

        out_filename = "_".join(out_filename)
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(output_subfolder, out_filename)
        shutil.copyfile(source_path, destination_path)

    elif "low" in filename and "NewYork" in filename:
        folder_name = "low_NewYork"
        out_filename = filename.split("_")[-3:]
        out_filename = "_".join(out_filename)
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(output_subfolder, out_filename)
        shutil.copyfile(source_path, destination_path)