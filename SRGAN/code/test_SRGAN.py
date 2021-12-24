from torch import nn
import time
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from netmodel_SRGAN import Generator, Discriminator, TruncatedVGG19
from dataset_SRGAN import SRDataset
import numpy as np
import PIL

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

# img_003_SRF_2_LR.png
# nr.jpg
# 测试图像
imgPath = '../../SRCNN/SRCNN-WBQ/data/my/nr.jpg'


if __name__ == '__main__':

    show=ToPILImage()

    scaling_factor = 2         # 放大比例

    # 生成器模型参数(与SRResNet相同)
    large_kernel_size_g = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size_g = 3   # 中间层卷积的核大小
    n_channels_g = 64         # 中间层通道数
    n_blocks_g = 16           # 残差模块数量

    checkpoint = '../model/SRGAN.pth'   # 预训练模型路径，如果不存在则为None

    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 模型初始化
    generator_GAN = Generator(large_kernel_size=large_kernel_size_g,
                                small_kernel_size=small_kernel_size_g,
                                n_channels=n_channels_g,
                                n_blocks=n_blocks_g,
                                scaling_factor=scaling_factor)

    SRGAN_model = torch.load(checkpoint)
    # start_epoch = SRGAN_model['epoch'] + 1
    generator_GAN.load_state_dict(SRGAN_model['generator'])

    generator_GAN.eval()
    model = generator_GAN

    # 加载图像
    img = Image.open(imgPath, mode='r')

    # pfy
    lr = np.array(PIL.Image.open(imgPath))
    h, w = lr.shape[0:2]
    nearest_img = nearest_interpolation(lr, h*2, w*2)
    nearest_img=Image.fromarray(nearest_img)

    img = img.convert('RGB')

    target_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5],std=[0.5]),
                        ])

    # 双线性上采样
    # Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    nearest_img.save('../result/nearest_img.jpg')

    # 图像预处理
    lr_img = target_transform(img)
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = show((sr_img+1)/2)
        sr_img.save('../result/SRGAN_img.jpg')

    print('用时  {:.3f} 秒'.format(time.time()-start))

    plt.subplot(1, 2, 1)
    plt.imshow(nearest_img)
    plt.title('input')
    plt.subplot(1, 2, 2)
    plt.imshow(sr_img)
    plt.title('output')
    plt.show()

