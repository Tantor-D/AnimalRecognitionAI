# 此文件用于将数据集修改为所需的形式并生成映射文件
# 在使用此代码前请务必记得修改if True/False这两个文件
import os
from PIL import Image

def renameDataSet(dataSetPath, mappingFilePath):
    # 重命名文件夹、生成datasetMapping.txt
    need_to_rename_dataset_and_gen_annotation = True
    need_to_rename_img_name = True

    dir_list = os.listdir(dataSetPath)
    dir_num = len(dir_list)
    if need_to_rename_dataset_and_gen_annotation:
        with open(mappingFilePath, mode="w", encoding='UTF-8') as f:
            for index, dir in enumerate(dir_list):
                # 写入映射映射文件
                f.write(("%03d" % (index + 1)) + " " + dir + "\n")
                # 开始进行文件夹命名
                old = dataSetPath + "/" + dir
                new = dataSetPath + "/" + ("%03d" % (index + 1))
                print(old + " " + new)
                os.rename(old, new)

    if need_to_rename_img_name:
        for i in range(1, dir_num + 1):
            folder_dir = dataSetPath + "/" + ("%03d" % i)
            file_list = os.listdir(folder_dir)
            for index, file_name in enumerate(file_list):
                if ".jpg" in file_name or ".png" in file_name or ".jpeg" in file_name:
                    old = folder_dir + "/" + file_name
                    new = folder_dir + "/" + ("%03d" % (index + 1)) + ".jpg"
                    im = Image.open(old)
                    im = im.convert('RGB')
                    im.save(new)
                    print(old + " " + new)
                    os.remove(old)

    print(dir_list)


# 然后进入每个文件夹之中对具体的图片文件进行重命名


if __name__ == "__main__":
    projectPath = "D:/Software_data/Pycharm_prj/AnimalRecognitionAI/"
    datasetPath = r"/buaa_animal_origin"    # 要修改的数据集
    mappingFilePath = "features/datasetMapping.txt"
    renameDataSet(projectPath + datasetPath, projectPath + mappingFilePath)
