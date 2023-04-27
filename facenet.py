import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.facenet import Facenet as facenet


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
# --------------------------------------------#
class Facenet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        # --------------------------------------------------------------------------#
        # todo 这个在上传服务器之后一定要改
        "model_path": r"D:\Software_data\Pycharm_prj\AnimalRecognitionAI\logs\1_resnet50_epoch=70_LFW=True\Epoch70-Total_Loss0.0064.pth-Val_Loss0.1541.pth",
        # --------------------------------------------------------------------------#
        #   输入图片的大小。
        # --------------------------------------------------------------------------#
        "input_shape": [224, 224, 3],
        # --------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络
        # --------------------------------------------------------------------------#
        "backbone": "resnet50",
        # --------------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

    def generate(self):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        model = facenet(backbone=self.backbone, mode="predict")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    # 本质上是一个不失真的resize，不会拉伸图片造成失真
    def letterbox_image(self, image, size):
        if self.input_shape[-1] == 1:
            image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # 仅仅用于predict的时候，会计算出两张图片的feature并进行欧式距离的计算
    def detect_image(self, image_1, image_2):
        # 图片预处理，归一化
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
            image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

            # np.expand_dims 添加batch_size维度
            # transpose将通道调整到第一维度，这样才能放进pytorch
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_1, np.float32) / 255, (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_2, np.float32) / 255, (2, 0, 1)), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # 图片传入网络进行预测，得到特征向量
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()

            # 计算二者之间的欧式距离
            l1 = np.linalg.norm(output1 - output2, axis=1)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        plt.show()
        return l1

    def cal_feature(self, image_1):
        # 图片预处理，归一化
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])

            # np.expand_dims 添加batch_size维度
            # transpose将通道调整到第一维度，这样才能放进pytorch
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_1, np.float32) / 255, (2, 0, 1)), 0))
            if self.cuda:
                photo_1 = photo_1.cuda()

            # 图片传入网络进行预测，得到特征向量
            output1 = self.net(photo_1).cpu().numpy()
        return output1
