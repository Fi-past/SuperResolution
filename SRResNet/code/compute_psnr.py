import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def sr_mse(src_img, dst_img):
    
    X = np.float32(src_img)
    Y = np.float32(dst_img)
    mse = np.mean(np.square(X-Y))
    return mse

def sr_pnsr(src_img, dst_img):

    mse = sr_mse(src_img, dst_img)
    range = 255
    pnsr = 10 * np.log10(range**2/mse)
    return pnsr

def sr_ssim(src_img, dst_img):
    X = np.float32(src_img)
    Y = np.float32(dst_img)
    mu1 = X.mean()
    mu2 = Y.mean()
    sigma1 = np.sqrt(((X - mu1) ** 2).mean())
    sigma2 = np.sqrt(((Y - mu2) ** 2).mean())
    sigma12 = ((X - mu1) * (Y - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def print_sr_value(src_img, dst_img):
    print("mse:{0}".format(sr_mse(src_img,dst_img)))
    print("pnsr:{0}".format(sr_pnsr(src_img, dst_img)))
    print("ssim:{0}".format(sr_ssim(src_img, dst_img)))


src = cv2.imread("../result/img_003_SRF_2_HR.png")
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# h, w = src.shape[0:2]

lr = cv2.imread("../result/bicubic.jpg")
lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

sr = cv2.imread("../result/srresnet.jpg")
sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)

plt.subplot(1,3,1)
plt.title('HR')
plt.imshow(src)

plt.subplot(1,3,2)
plt.title('LR')
plt.imshow(lr)

plt.subplot(1,3,3)
plt.title('SR')
plt.imshow(sr)

print_sr_value(src, sr)

plt.show()
