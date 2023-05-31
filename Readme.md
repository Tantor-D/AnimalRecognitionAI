# Animal Recognition AI

## 简要说明

本项目为魔改faceNet得到的动物个体识别系统，就是让动物体想 = 人脸 然后来进行处理。

## 一些重要文件

**文件夹：**

- `buaa_animal_origin`：存的是进行预测时，
- `datasets`：存的是训练集，一个文件夹中的图片为同一个label
- `lfw`：存的是测试集，一个文件夹中的图片为同一个label
- `features`：程序训完后，进行预测时需要用到的文件在里面
  - `cls_feature.txt`：里面存了预测图片是需要对比的那一堆数据的label和路径，然后利用这个文件的信息进行特征生成
  - `features.npy`：
  - `labels.txt`：
- `logs`：
- `model_data`：
- `nets`：
- `utils`：

**文件：**

- `gen_annotation.py`：进行训练前需要利用这个文件生成cls_train.txt，用于生成label和图片的路径
- `cls_train.txt`：里面存着每张图片的绝对地址和label
- `eval_LFW`：训练出的模型只能用于生成features，利用`lfw`这个路径下的数据集生成测试时要设置的距离阈值和计算测试集上最高的准确率



## 使用的步骤

1. 使用`utils/renameDataset.py` 将数据集的命名方式修改到合格的形式，同时生成“文件夹-动物名”的map

2. 使用`gen_features.py` 来用模型跑一遍数据后台的图片，用于之后匹配的距离计算
