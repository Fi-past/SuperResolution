from os import listdir
from os.path import join

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import PIL.Image as pil_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    img, _, _ = img.split()
    return img


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, zoom_factor, crop_size):
        super(DatasetFromFolder, self).__init__()
        # for x in listdir(image_dir):
        #     print(x)
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.preTrans=transforms.Compose([
                                # transforms.RandomCrop(crop_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                # transforms.RandomRotation(30),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                ])

        self.input_transform = transforms.Compose([
                                #transforms.CenterCrop(crop_size),
                                # transforms.Resize(crop_size//zoom_factor),
                                # transforms.Resize(crop_size, interpolation=Image.BICUBIC),  # 双线性插值 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5]),
                                ])

        self.target_transform = transforms.Compose([
                                #transforms.CenterCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5]),
                                ])


    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        input=self.preTrans(input)

        # print(input.size)

        height = (input.size[0]//2)*2
        width = (input.size[1]//2)*2


        target = input.copy()
        
        input=input.resize((height//2,width//2))
        input=input.resize((height,width))
        input = self.input_transform(input)
        target=target.resize((height,width))
        target = self.target_transform(target)

        # print(target.size())

        return input, target

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    # isImageFile=is_image_file('./data/test.jpeg')
    # print(isImageFile)
    imgdataset=DatasetFromFolder('./data/train',2,224)
    show=ToPILImage()
    (input, target) = imgdataset[10]
    input=show((input+1)/2)
    target=show((target+1)/2)

    plt.subplot(1, 2, 1)
    plt.imshow(input)
    plt.title('input')
    plt.subplot(1, 2, 2)
    plt.imshow(target)
    plt.title('target')

    plt.show()