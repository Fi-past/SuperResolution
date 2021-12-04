import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 准备修改一下，仿照那个作者的模式
# crop_size = 224
imgPath='F:\\Desktop\\All\\language\\VSCODEWORKSPACE\\Python\\Class\\计算机视觉\\SuperResolution\\SRCNN\\SRCNN-WBQ\\data\\my\\1.jpg'
model_path='./model/FSRCNN_model_torch1_4_25_8.pth'

with torch.no_grad():
    show=ToPILImage()

    target_transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])

    img = Image.open(imgPath).convert('YCbCr')

    height = (img.size[0]//2)*2
    width = (img.size[1]//2)*2

    y, cb, cr = img.split()

    # cb=cb.resize((height,width))
    # cr=cr.resize((height,width))
    # y=y.resize((height//2,width//2))

    RSCNN_model = torch.load(model_path,map_location='cpu')
    RSCNN_model.eval()

    input_y = target_transform(y).view(1, -1, y.size[1], y.size[0])  # we only work with the "Y" channel
    out_y = RSCNN_model(input_y)
    out_y=show(torch.squeeze(out_y))

    cb = target_transform(cb).view(1, -1, cb.size[1], cb.size[0])  
    cb = RSCNN_model(cb)
    cb=show(torch.squeeze(cb))

    cr = target_transform(cr).view(1, -1, cr.size[1], cr.size[0])  
    cr = RSCNN_model(cr)
    cr=show(torch.squeeze(cr))


    out_img = Image.merge('YCbCr', [out_y, cb, cr]).convert('RGB')
    

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
    plt.title('target')
    plt.show()
