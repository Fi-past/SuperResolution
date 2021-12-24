from torch import nn
import time
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from netmodel_EDSR import EDSR

# 测试图像
# hi.jpg
# img_003_SRF_2_LR.png

imgPath = '../../SRCNN/SRCNN-WBQ/data/my/img_003_SRF_2_LR.png'

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 2      # 放大比例

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    show=ToPILImage()
    # 预训练模型
    SRResNet_model_path = "../model/EDSR_26_5.pth"

    # 加载模型SRResNet 或 SRGAN
    SRResNet = torch.load(SRResNet_model_path)
    SRResNet.eval()
    model = SRResNet

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    target_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5],std=[0.5]),
                        ])

    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    Bicubic_img.save('../result/bicubic.jpg')

    # 图像预处理
    lr_img = target_transform(img)
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = show((sr_img+1)/2)
        sr_img.save('../result/srresnet.jpg')

    print('用时  {:.3f} 秒'.format(time.time()-start))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('input')
    plt.subplot(1, 2, 2)
    plt.imshow(sr_img)
    plt.title('output')
    plt.show()

