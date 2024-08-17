import os
import random
import shutil
import argparse

def split_datasets(input_path, output_path,data_name, block_name,t):
    # res = ['high', 'mid', 'low']

    split = ['pretrain', 'finetune', 'test']
    
    root_annotations_path = os.path.join(input_path, "Annotations")
    root_images_path = os.path.join(input_path, "images")
    root_instances_path = os.path.join(input_path, "Instance_masks")
    root_semantic_path = os.path.join(input_path, "Semantic_masks")

    for res_name in os.listdir(root_images_path):
        res_path = os.path.join(root_images_path, res_name)
        # 创建列表用于存储图片文件名，str类型
        images_list = []
        if os.path.isdir(res_path):
            for img_name in os.listdir(res_path):
                if img_name.endswith(".jpg"):
                    images_list.append(img_name[:-4])
        # 将列表顺序打乱
        random.shuffle(images_list)
        print("图片名称列表读取完毕！")

        # 将打乱后的列表按4:4:2的比例划分为训练集、验证集和测试集，列表切片需整数
        images_size = len(images_list)
        images_train = images_list[0:round(images_size * 0.4)]
        images_val = images_list[round(
            images_size * 0.4):round(images_size * 0.8)]
        images_test = images_list[round(images_size * 0.8):]
        # print(len(images_train), len(images_val), len(images_test))

        # 判断目标目录是否存在train、val、test三个文件夹，不存在则在目标目录下创建train、val、test三个文件夹
        target__train_path = os.path.join(output_path, split[0],t,res_name)
        target__val_path =  os.path.join(output_path, split[1],t,res_name)
        target__test_path =  os.path.join(output_path, split[2],t,res_name)

        tmp=["Annotations","RS_images","Instance_masks","Semantic_masks"]
        for i in tmp:
            if os.path.exists(os.path.join(target__train_path,i,data_name,block_name)) == False:
                os.makedirs(os.path.join(target__train_path,i,data_name,block_name))

            if os.path.exists(os.path.join(target__val_path,i,data_name,block_name)) == False:
                os.makedirs(os.path.join(target__val_path,i,data_name,block_name))

            if os.path.exists(os.path.join(target__test_path,i,data_name,block_name)) == False:
                os.makedirs(os.path.join(target__test_path,i,data_name,block_name))

            if i == "RS_images":
                suffixes = ['.txt', '.caption', '.jpg']
                for suffix in suffixes:
                    for image_train in images_train:
                        if os.path.isfile(os.path.join(root_images_path, res_name,image_train + suffix)):
                            shutil.copy(os.path.join(root_images_path, res_name,image_train + suffix), os.path.join(target__train_path,i,data_name,block_name))
                            print(f"已复制文件: {os.path.join(root_images_path, res_name,image_train + suffix)}")
                        else:
                            print(f"文件不存在: {os.path.join(root_images_path, res_name,image_train + suffix)}")
                    for image_val in images_val:
                        if os.path.isfile(os.path.join(root_images_path, res_name,image_val + suffix)):
                            shutil.copy(os.path.join(root_images_path, res_name,image_val + suffix), os.path.join(target__val_path,i,data_name,block_name))
                            print(f"已复制文件: {os.path.join(root_images_path, res_name,image_val + suffix)}")
                        else:
                            print(f"文件不存在: {os.path.join(root_images_path, res_name,image_val + suffix)}")
                    for image_test in images_test:
                        if os.path.isfile(os.path.join(root_images_path,res_name, image_test + suffix)):
                            shutil.copy(os.path.join(root_images_path, res_name,image_test + suffix), os.path.join(target__test_path,i,data_name,block_name))
                            print(f"已复制文件: {os.path.join(root_images_path,res_name, image_test + suffix)}")
                        else:
                            print(f"文件不存在: {os.path.join(root_images_path,res_name, image_test + suffix)}")
            if i == "Annotations":
                suffix = '.json'
                for image_train in images_train:
                    if os.path.isfile(os.path.join(root_annotations_path,res_name, image_train + suffix)):
                        shutil.copy(os.path.join(root_annotations_path, res_name,image_train + suffix),
                                    os.path.join(target__train_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_annotations_path, res_name,image_train + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_annotations_path, res_name,image_train + suffix)}")
                for image_val in images_val:
                    if os.path.isfile(os.path.join(root_annotations_path, res_name,image_val + suffix)):
                        shutil.copy(os.path.join(root_annotations_path, res_name,image_val + suffix), os.path.join(target__val_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_annotations_path, res_name,image_val + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_annotations_path, res_name,image_val + suffix)}")
                for image_test in images_test:
                    if os.path.isfile(os.path.join(root_annotations_path, res_name,image_test + suffix)):
                        shutil.copy(os.path.join(root_annotations_path, res_name,image_test + suffix), os.path.join(target__test_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_annotations_path, res_name,image_test + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_annotations_path, res_name,image_test + suffix)}")
            if i=="Instance_masks":
                suffix = '.png'
                for image_train in images_train:
                    if os.path.isfile(os.path.join(root_instances_path,res_name,image_train + suffix)):
                        shutil.copy(os.path.join(root_instances_path, res_name,image_train + suffix), os.path.join(target__train_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_instances_path, res_name,image_train + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_instances_path,res_name, image_train + suffix)}")
                for image_val in images_val:
                    if os.path.isfile(os.path.join(root_instances_path, res_name,image_val + suffix)):
                        shutil.copy(os.path.join(root_instances_path, res_name,image_val + suffix), os.path.join(target__val_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_instances_path,res_name, image_val + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_instances_path, res_name,image_val + suffix)}")
                for image_test in images_test:
                    if os.path.isfile(os.path.join(root_instances_path, res_name,image_test + suffix)):
                        shutil.copy(os.path.join(root_instances_path, res_name,image_test + suffix), os.path.join(target__test_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_instances_path, res_name,image_test + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_instances_path,res_name, image_test + suffix)}")
            if i=="Semantic_masks":
                suffix = '.png'
                for image_train in images_train:
                    if os.path.isfile(os.path.join(root_semantic_path, res_name,image_train + suffix)):
                        shutil.copy(os.path.join(root_semantic_path, res_name,image_train + suffix), os.path.join(target__train_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_semantic_path, res_name,image_train + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_semantic_path, res_name,image_train + suffix)}")
                for image_val in images_val:
                    if os.path.isfile(os.path.join(root_semantic_path, res_name,image_val + suffix)):
                        shutil.copy(os.path.join(root_semantic_path, res_name,image_val + suffix), os.path.join(target__val_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_semantic_path,res_name, image_val + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_semantic_path, res_name,image_val + suffix)}")
                for image_test in images_test:
                    if os.path.isfile(os.path.join(root_semantic_path,res_name, image_test + suffix)):
                        shutil.copy(os.path.join(root_semantic_path,res_name, image_test + suffix), os.path.join(target__test_path, i,data_name,block_name))
                        print(f"已复制文件: {os.path.join(root_semantic_path, res_name,image_test + suffix)}")
                    else:
                        print(f"文件不存在: {os.path.join(root_semantic_path,res_name, image_test + suffix)}")
        print("数据集分割完毕！")


# 创建一个ArgumentParser对象
parser = argparse.ArgumentParser(description='Process some integers.')

# 添加参数，这里的dest是在代码中引用这个参数时使用的变量名
parser.add_argument('--data_path', dest='folder_path', default='/data/dailei/WHUGeoGen_v2/RAW/Beijing',type=str, help='input subdataset path like /home/wyd/mxq/data/OSM2/13Buffalo')
parser.add_argument('--output_path', dest='output_path', default='/data/dailei/WHUGeoGen_v2/RAW/Beijing/split',type=str, help='input subdataset path like /home/wyd/mxq/data/OSM2/13Buffalo')
parser.add_argument('--data_name', dest='data_name', default='BJ11_21_new',type=str, help='input subdataset name like BeiJing')
parser.add_argument('--block_name', dest='block_name', default='BJ11',type=str, help='input subdataset name like BJ1')

# 解析命令行参数
args = parser.parse_args()

folder_path = args.folder_path # 根文件夹路径
output_path = args.output_path # 输出文件夹路径
data_name = args.data_name # 数据集名
block_name = args.block_name # 块名
# ratio = float(args.ratio) # 过滤标签的比例
# type = ['512','1024','2048','focus']
# type = ['data512', 'data1024', 'data2048']
type = ['data768', 'data1280']

# block_name="NewYork"
# 21Olive
for t in type:
    s_path = os.path.join(folder_path, data_name, block_name,t,"data")
    # d_path=os.path.join(output_path, input_path)
    split_datasets(s_path,output_path,data_name, block_name,t)
        # folder_path = "/home/wyd/mxq/data/OSM2/11SaratogaSprings" + "/" + t +"/images/" + r

# tar -cvf - finetune pretrain test | pigz -p $(nproc) > WHUGeoGen_v2.tar
