import paddle
from paddle.vision import transforms
import PIL.Image as Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import PIL
# img_003_SRF_2_LR.png
# nr.jpg
# 1.jpg
# gauss100075.jpg
# gaussadd (18).jpg

def nearest_interpolation(src_img, dstH, dstW):
    srcH, srcW, channel = src_img.shape
    dst_img = np.zeros((dstH, dstW, channel), dtype=np.uint8)
    for k in range(channel):
        for i in range(dstH):
            for j in range(dstW):
                x = min(round(i * (srcH/dstH)), srcH-1)
                y = min(round(j * (srcH/dstH)), srcW-1)
                dst_img[i, j, k] = src_img[x, y, k]
    return dst_img


img_path='F:/Desktop/All/language/VSCODEWORKSPACE/Python/Class/计算机视觉/SuperResolution/SRCNN/SRCNN-WBQ/data/my/img_003_SRF_2_LR.png'
img=Image.open(img_path,mode="r")  #选择自己图片的路径

# pfy
lr = np.array(PIL.Image.open(img_path).convert('RGB'))
h, w = lr.shape[0:2]

# nearest_img = nearest_interpolation(lr, h*2, w*2)
nearest_img = lr

imgT=transforms.ToTensor()(img).unsqueeze(0)

#导入模型
net=paddle.jit.load("../paddle/X1/edsr_paddle")

source = net(imgT)[0, :, :, :]
source = source.cpu().detach().numpy()  # 转为numpy
source = source.transpose((1, 2, 0))  # 切换形状
source = np.clip(source, 0, 1)  # 修正图片
source = Image.fromarray(np.uint8(source * 255))

nearest_img=Image.fromarray(nearest_img)

plt.subplot(1, 2, 1)
plt.imshow(nearest_img)
plt.title('input')
plt.subplot(1, 2, 2)
plt.imshow(source)
plt.title('output')
plt.show()

# img.show()
nearest_img.save('../result/lr_paddle.jpg')
source.save('../result/sr_paddle.jpg')