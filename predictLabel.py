# 需要输入一个图片的path，随后预测出这只动物的label
# 输出的是一个label，按照置信度进行排序，置信度的唯一判断标准是特征距离低于阈值的图片的数量占训练集中数量的百分比（后面要改，因为实际使用中的话训练集中的图片数量不一样）
# 没有图片匹配的话返回一个-1
import argparse
import os.path
import sys

from PIL import Image
from facenet import Facenet
import numpy as np


# 默认的阈值距离为超参数，要根据训练的结果进行调整
def predictLabel(img_path, mode, threshold_dis=1.10000):
    # 说明没有指定图片的地址
    if len(img_path) == 0:
        writeToTxt("-1", mode)  # 约定-1表示空地址
        return -1
    if not os.path.exists(img_path):
        writeToTxt("-2", mode)  # 约定-2表示图片不存在
        return -2

    # 准备好feature，labels和模型
    model = Facenet()

    # todo 也许需要进行异常处理

    project_path = r"D:/Software_data/Pycharm_prj/AnimalRecognitionAI/" if mode == 'local' else r"/root/AnimalRecognitionAI/"
    features = np.load(project_path + "features/features.npy")
    with open(project_path + "labels.txt") as f:
        dataset_labels = f.readlines()
    labels = []
    for label in dataset_labels:
        labels.append(int(label))
    num_classes = np.max(labels) + 1

    # 每个label中图片的数量和
    img_num_label_index = [0 for i in range(num_classes)]

    for label in labels:
        img_num_label_index[label] = img_num_label_index[label] + 1

    # 准备好图片开始计算
    img = Image.open(img_path)
    img_feature = model.cal_feature(img)
    correct_num_label_index = [0 for i in range(num_classes)]  # 距离在阈值内的图片的数量

    # 进行计算
    for i, label in enumerate(labels):
        feature = features[i]  # 训练集中图片的feature
        dis = np.linalg.norm(feature - img_feature, axis=1)
        # print(f"index={i}, label={label}, dis={dis}")
        if dis < threshold_dis:
            # print("i am in")
            correct_num_label_index[label] = correct_num_label_index[label] + 1
            # print(f"correct_num={correct_num_label_index[label]}")

    # 根据比例找出最有可能的label
    max_correct_rate = 0
    correct_label = -3
    for i in range(num_classes):
        if (correct_num_label_index[i] > 0):
            rate = correct_num_label_index[i] / img_num_label_index[i]

            if rate > max_correct_rate:
                max_correct_rate = rate
                correct_label = i

    print(f"model predict label:{correct_label}")
    print(f"it means real label:{correct_label + 1}")
    return correct_label


def writeToTxt(msg, mode):
    predict_txt_path = "/root/AnimalManagement/temp/predictLabel.txt" if mode == "server" else "C:/Users/Tantor/Desktop/predictLabel.txt"
    predict_txt = open(predict_txt_path, mode="w")
    predict_txt.write(msg)
    predict_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ai detect')
    parser.add_argument('--img_path', default="", type=str, help='input img path')
    parser.add_argument('--mode', default="local", type=str, choices=["local", "server"],help='input run type,local or server')
    args = parser.parse_args()

    # 开始正式预测
    label = predictLabel(args.img_path, args.mode)

    # 有一个要注意的地方，预测出的label+1 才是对应的文件夹的命名，此处需要写入文件的是文件夹的命名，因此要+1
    # 将最终的结果写入预先定好的txt文件中，只有当返回值大于等于0时才说明运行正常，等于-3说明没有匹配的动物
    if label >= 0:
        writeToTxt(str(label + 1), args.mode)
    elif label == -3:
        writeToTxt("-3", args.mode)
    else:
        writeToTxt("-4", args.mode)  # 出现-4说明有问题
