# 此文件用于训练完之后生成训练集中图片的特征
# 使用此文件之前请务必生成了正确的cls_train.txt（准确的是说把文件的结构改好

from PIL import Image
from facenet import Facenet
import numpy as np
import os


def gen_feature_annotation(cls_path, datasets_path):
    # 可调参数，要比较的那个数据集的信息

    types_name = os.listdir(datasets_path)
    types_name = sorted(types_name)

    list_file = open(cls_path, 'w')
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            list_file.write(
                str(cls_id) + ";" + '%s' % (os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()


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
    print(features.shape)
    np.save("features/features.npy", features)
    return


if __name__ == "__main__":
    # 首先生成cls_features文件，里面存了label和路径，然后利用这个文件的信息进行特征生成
    annotation_path = "features/cls_feature.txt"
    feature_dataset = "buaa_cat_name/"

    gen_feature_annotation(annotation_path, feature_dataset)
    cal_feature(annotation_path)
