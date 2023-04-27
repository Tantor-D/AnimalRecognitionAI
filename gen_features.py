# 此文件用于训练完之后生成训练集中图片的特征

from PIL import Image
from facenet import Facenet
import numpy as np


def cal_feature(annotation_path):
    # 得到训练集图片路径的信息
    with open(annotation_path) as f:
        dataset_path = f.readlines()
    labels = []  # 存的是label的信息
    paths = []  # 存的是图片的绝对地址
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
        paths.append(path_split[1])

    # 写入lable信息到labels.txt中
    label_txt = open('features/labels.txt', mode='w')
    for label in labels:
        label_txt.write(str(label) + "\n")
    label_txt.close()

    # 进行特征的计算和写入
    model = Facenet()
    for index, img_path in enumerate(paths):
        img = Image.open(img_path[0:-1])
        img_feature = model.cal_feature(img)
        features = img_feature if index == 0 else np.vstack((features, img_feature))
        # print(features.shape)
    np.save("features/features.npy", features)
    return


if __name__ == "__main__":
    annotation_path = "cls_train.txt"
    cal_feature(annotation_path)
