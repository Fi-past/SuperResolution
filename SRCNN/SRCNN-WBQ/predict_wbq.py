import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 准备修改一下，仿照那个作者的模式
# crop_size = 224
imgPath='./data/my/gauss100075.jpg'
model_path='./model/SRCNN_model_torch14_24.pth'

with torch.no_grad():
    show=ToPILImage()

    target_transform = transforms.Compose([
                        # transforms.Resize([crop_size, crop_size]),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
                        ])

    img = Image.open(imgPath).convert('YCbCr')
    y, cb, cr = img.split()

    input = target_transform(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel

    RSCNN_model = torch.load(model_path,map_location='cpu')
    RSCNN_model.eval()

    out = RSCNN_model(input)
    out=show(torch.squeeze(out))
    out_img = Image.merge('YCbCr', [out, cb, cr]).convert('RGB')
    

    input=img
    target=out_img

    # input=show((img_tensor+1)/2)
    # target=show((out+1)/2)

    # 深色区域会有蓝色部分 是训练不充分的问题
    plt.subplot(1, 2, 1)
    plt.imshow(input)
    plt.title('input')
    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.title('output')
    plt.show()
