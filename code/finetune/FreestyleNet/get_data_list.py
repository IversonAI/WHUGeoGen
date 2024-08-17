import os

all_file_full_path_list = []
all_file_name_list = []

def get_all_files(path):
    """
    获取指定路径下多层目录内的所有文件全路径及文件名称
    :param path: 需获取文件的指定路径
    :return: 结果1 类型：list<str> ：多层目录下的，全部文件全路径；结果2 类型：list<str> ：多层目录下的，全部文件名称
    """

    all_file_list = os.listdir(path)
    # 遍历该文件夹下的所有目录或文件
    for file in all_file_list:
        file_path = os.path.join(path, file)
        # 如果是文件夹，递归调用当前函数
        if os.path.isdir(file_path):
            get_all_files(file_path)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(file_path):
            if file_path.endswith('.jpg'):
                all_file_full_path_list.append(file_path)
                all_file_name_list.append(file)
    return all_file_full_path_list, all_file_name_list


# print(getallfile('C:/Users/ymt30/Desktop/temp/'))
#print(path_read)

path_list1, file_name_list1 = get_all_files('/data/dailei/WHUGeoGen3/test/data512')
# file1 = open('/data/dailei/FLAIR/FLAIR_val.txt', 'w')
f1=open("/data/dailei/WHUGeoGen3/test/data512/WHUGeoGenv2_val.txt", 'w')
for i in path_list1:
    f1.write(i + '\n')
f1.close()

all_file_full_path_list = []
all_file_name_list = []

path_list2, file_name_list2 = get_all_files('/data/dailei/WHUGeoGen3/finetune/data512')
f2=open("/data/dailei/WHUGeoGen3/finetune/data512/WHUGeoGenv2_train.txt", 'w')
for i in path_list2:
    f2.write(i + '\n')
f2.close()


# use linux shell
# find . -type f -print > file_list.txt