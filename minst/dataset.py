import paddle
import json
import gzip
import numpy as np


# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = '../data/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        if mode == 'train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode == 'valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode == 'eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")

        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))

        # 全连接网络
        self.imgs = imgs
        self.labels = labels
        # 卷积网络
        # self.imgs = imgs.reshape(1,28,28)
        # self.labels = labels.reshape(1,28,28)

    def __getitem__(self, idx):
        # 全连接网络
        # img = np.array(self.imgs[idx]).astype('float32')
        # label = np.array(self.labels[idx]).astype('float32')
        # 卷积网络
        img = np.array(self.imgs[idx]).astype('float32').reshape([1,28,28])
        label = np.array(self.labels[idx]).astype('float32').reshape([1])

        return img, label

    def __len__(self):
        return len(self.imgs)