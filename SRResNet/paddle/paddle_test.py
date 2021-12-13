import paddle
from paddle.vision import transforms
import PIL.Image as Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img=Image.open('F:/Desktop/All/language/VSCODEWORKSPACE/Python/Class/计算机视觉/SuperResolution/SRCNN/SRCNN-WBQ/data/my/img_003_SRF_4_LR.png',mode="r")  #选择自己图片的路径
imgT=transforms.ToTensor()(img).unsqueeze(0)

#导入模型
net=paddle.jit.load("./ref/example.dy_model2/linear")

source = net(imgT)[0, :, :, :]
source = source.cpu().detach().numpy()  # 转为numpy
source = source.transpose((1, 2, 0))  # 切换形状
source = np.clip(source, 0, 1)  # 修正图片
source = Image.fromarray(np.uint8(source * 255))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('input')
plt.subplot(1, 2, 2)
plt.imshow(source)
plt.title('output')
plt.show()

# img.show()
img.save('../result/lr_paddle.jpg')
source.save('../result/sr_paddle.jpg')