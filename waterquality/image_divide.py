import os
import random
import cv2
import numpy as np
from PIL import Image

path = r"E:/PythonProject/venv/Tongxing/data/data3/val/2/"
path0 = r"E:/PythonProject/venv/Tongxing/data/data3/val/0/"
for filename in os.listdir(path):
    #原图
    img = cv2.imread(path + '/' + filename)
    height, width = img.shape[:2]
    print(img.shape)

    w = int(width/2)
    h = height

    img1 = np.ones((h, w, 3), dtype=np.uint8) * 255
    img2 = np.ones((h, width-w, 3), dtype=np.uint8) * 255

    img1[:h,:w,:] = img[:h,:w,:]
    img2[:h,:width-w,:] = img[:h,w:width,:]

    #cv2.namedWindow("imgs", cv2.WINDOW_NORMAL)
    #imgs = np.hstack([img1, img2, img3, img4])
    #cv2.imshow("imgs", dst)
    cv2.imwrite(path0 + '0'+filename, img1)
    cv2.imwrite(path0 + '1'+filename, img2)

    #print(imgs.shape)
    #cv2.waitKey(0)