"""
利用递归实现目录的遍历
@para  sourcePath:原文件目录
@para  targetPath:目标文件目录
"""
import os
import shutil

# # -*- coding:utf-8 -*-
# import os
# import shutil
#
#
# def copyFile(sourcePath, savePath):
#     image_end = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG']
#     json_end = ['.json', '.JSON']
#     for dir_or_file in os.listdir(sourcePath):
#         filePath = os.path.join(sourcePath, dir_or_file)
#         if os.path.isfile(filePath):  # 判断是否为文件
#             # 如果文件是图片，则复制，如果都是同一种图片类型也可以用这句：if os.path.basename(filePath).endswith('.jpg'):
#             if os.path.splitext(os.path.basename(filePath))[1] in image_end:
#                 print('this copied pic name is ' + os.path.basename(filePath))  # 拷贝jpg文件到自己想要保存的目录下
#                 shutil.copyfile(filePath, os.path.join(savePath, os.path.basename(filePath)))
#             else:
#                 continue
#         elif os.path.isdir(filePath):  # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
#             copyFile(filePath, savePath)
#         else:
#             print('not file and dir ' + os.path.basename(filePath))
#
# # 2048:695
# # 1024:2862
# # 512:11430
# # 256:45385
#
# if __name__ == '__main__':
#     sourcePath = "/123"
#     savePath = "/567"
#     copyFile(sourcePath, savePath)

import os


def ListFilesToTxt(dir, file, wildcard, level_i,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    (filename, extension) = os.path.splitext(name)
                    if filename.startswith(str(level_i)):
                        file.write(filename  + "\n")
                    break


def Test():
    dir = "/media/wyd-7/ba7c6ce2-33ae-42f2-9777-a6aa32ab2e66/TEST_NY/output_cu_256/images"

    level = [15,16,17,18,19,20]
    for i in level:
        # print(i)
        outfile=f'/media/wyd-7/ba7c6ce2-33ae-42f2-9777-a6aa32ab2e66/TEST_NY/output_cu_256/files_level{i}.txt'
    # outfile = "/media/wyd-7/ba7c6ce2-33ae-42f2-9777-a6aa32ab2e66/TEST_NY/output_cu_2048/image_name.txt"
        wildcard = ".jpg"

        file = open(outfile, "w")
        if not file:
            print("cannot open the file %s for writing" % outfile)
        ListFilesToTxt(dir, file, wildcard,i, 1)

        file.close()


Test()