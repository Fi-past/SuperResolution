import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt



class SRDataset(Dataset):

    def __init__(self, data_path, crop_size, scaling_factor):
        """
        :参数 data_path: 图片文件夹路径
        :参数 crop_size: 高分辨率图像裁剪尺寸  （实际训练时不会用原图进行放大，而是截取原图的一个子块进行放大）
        :参数 scaling_factor: 放大比例
        """

        self.data_path=data_path
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.images_path=[]

        # 如果是训练，则所有图像必须保持固定的分辨率以此保证能够整除放大比例
        # 如果是测试，则不需要对图像的长宽作限定

        # 读取图像路径
        for name in os.listdir(self.data_path):
            self.images_path.append(os.path.join(self.data_path,name))

        # 数据处理方式
        self.pre_trans=transforms.Compose([
                                # transforms.CenterCrop(self.crop_size),
                                transforms.RandomCrop(self.crop_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
                                ])

        self.input_transform = transforms.Compose([
                                transforms.Resize(self.crop_size//self.scaling_factor),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5]),
                                ])

        self.target_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5]),
                                ])


    def __getitem__(self, i):
        # 读取图像
        img = Image.open(self.images_path[i], mode='r')
        img = img.convert('RGB')
        img=self.pre_trans(img)

        lr_img = self.input_transform(img)
        hr_img = self.target_transform(img.copy())
        

        return lr_img, hr_img


    def __len__(self):
        return len(self.images_path)


# 单元测试通过
def main():
    train_path='../../SRCNN/SRCNN-WBQ/data/train'
    test_path='../../SRCNN/SRCNN-WBQ/data/test'
    ds=SRDataset(train_path,224,2)
    l,h=ds[0]
    print(l.shape)
    print(h.shape)



if __name__ == '__main__':
    main()
    