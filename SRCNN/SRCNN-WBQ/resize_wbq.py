import cv2

root_path='./data/my/'
name='100075.jpg'

img=cv2.imread(root_path+name)
out=cv2.GaussianBlur(img,(7,7),1.2)
cv2.imwrite(root_path+'gauss'+name,out)