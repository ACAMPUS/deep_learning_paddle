import numpy as np
from paddle.io import Dataset


# 构建一个类，继承paddle.io.Dataset，创建数据读取器
# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         # 样本数量
#         self.num_samples = num_samples
#
#     def __getitem__(self, idx):
#         # 随机产生数据和label
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1,)).astype('float32')
#         return image, label
#
#     def __len__(self):
#         # 返回样本总数量
#         return self.num_samples
#
#
# # 测试数据读取器
# dataset = RandomDataset(10)
# for i in range(len(dataset)):
#     print(dataset[i])


import paddle
import json
import gzip
import numpy as np


# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './work/mnist.json.gz'
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

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx]).astype('float32')
        label = np.array(self.labels[idx]).astype('float32')

        return img, label

    def __len__(self):
        return len(self.imgs)


# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
train_dataset = MnistDataset(mode='train')
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)

