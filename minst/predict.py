# 读取一张本地的样例图片，转变成模型输入的格式
import numpy as np
import paddle
from PIL import Image

# 导入图像读取第三方库
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from minst.model.conv import MNIST

img_path = '../test/1.jpg'
# 读取原始图像并显示
im = Image.open(img_path)
print(type(im))
plt.imshow(im)
plt.show()
# 将原始图像转为灰度图
im = im.convert('L')
print('原始图像shape: ', np.array(im).shape)
# 使用Image.ANTIALIAS方式采样原始图片
im = im.resize((28, 28), Image.ANTIALIAS)
plt.imshow(im)
plt.show()
print("采样后图片shape: ", np.array(im).shape)


def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # 图像归一化
    im = 1.0 - im / 255.
    return im


# 定义预测过程
model = MNIST()
params_file_path = 'mnist.pdparams'
img_path = '../test/1.jpg'
# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = load_image(img_path)
result = model(paddle.to_tensor(tensor_img))
print('result', result)
#  预测输出取整，即为预测的数字，打印结果
print("本次预测的数字是", result.numpy().astype('int32'))
