import os

# 设置源文件夹路径和目标文件路径
source_folder = '/data/dailei/WHUGeoGen3/sum'
target_file = '/data/dailei/WHUGeoGen3/output_new.txt'

# 打开目标文件以供写入
with open(target_file, 'w') as outfile:
    # 获取文件夹中所有txt文件
    txt_files = [f for f in os.listdir(source_folder) if f.endswith('.txt')]
    sorted_files = sorted(txt_files)
    # 遍历每个txt文件
    for txt_file in txt_files:
        # 获取文件的完整路径
        file_path = os.path.join(source_folder, txt_file)

        # 读取文件内容并写入目标文件
        with open(file_path, 'r') as infile:
            content = infile.read()
            outfile.write(f'{txt_file[:-4]}:{content}')