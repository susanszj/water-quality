import cv2
import numpy as np
import scipy.signal as ss
import os
import matplotlib.pyplot as plt

filePath0 = "E:/PythonProject/venv/Tongxing/data/data3"
filePath1 = "E:/PythonProject/venv/Tongxing/data0"

path_dirs = "C:/Users/25005/Desktop/724/1"
path_0 = "E:/PythonProject/venv/Tongxing/data/river_filter/1"

for filename in os.listdir(path_dirs):
    path0 = path_dirs + '/' + filename
    path1 = path_0 + "/" + filename
    if filename not in path_0:
        print(path0)
        img = cv2.imread(path0).astype(float) / 255
        rand = np.random.rand(*img.shape)
        rand = rand * (rand > 0.9)
        img += rand

        img1 = img + 0  # 深拷贝
        for i in range(3):
            img1[:, :, i] = ss.medfilt2d(img1[:, :, i], [5, 5])
        img1 = (img1 * 255).astype(np.uint8)
        img2 = cv2.pyrMeanShiftFiltering(img1, 25, 25)
        cv2.imwrite(path1, img2)
'''
for oneDir in os.listdir(filePath0):
    path_oneDir = filePath0+'/'+oneDir
    for dirs in os.listdir(path_oneDir):
        path_dirs = path_oneDir+'/'+dirs
        for filename in os.listdir(path_dirs):
            path0 = path_dirs + '/'+filename
            path1 = filePath1+'/'+oneDir+'/'+dirs+'/'+filename
            print(path0)
            dir_path = filePath1+'/'+oneDir+'/'+dirs
            if filename not in dir_path:
                img = cv2.imread(path0).astype(float)/255
                rand = np.random.rand(*img.shape)
                rand = rand * (rand > 0.9)
                img += rand

                img1 = img + 0  # 深拷贝
                for i in range(3):
                    img1[:, :, i] = ss.medfilt2d(img1[:, :, i], [5, 5])
                img1 = (img1 * 255).astype(np.uint8)
                img2 = cv2.pyrMeanShiftFiltering(img1, 25, 25)
                cv2.imwrite(path1, img2)

'''
'''
img = cv2.imread(r'E:\PythonProject\Tongxing\data\train\1\XwwzB.jpg').astype(float)/255
rand = np.random.rand(*img.shape)
rand = rand * (rand > 0.9)
img += rand


img1 = img + 0  #深拷贝
for i in range(3):
    img1[:,:,i] = ss.medfilt2d(img1[:,:,i], [5,5])
img1 = (img1 * 255).astype(np.uint8)
img2=cv2.pyrMeanShiftFiltering(img1,25,25)

cv2.imshow('mean1',img1)
cv2.imwrite('1.png',img1)
cv2.imshow('mean2',img2)
cv2.waitKey()
cv2.destroyAllWindows()'''


'''
img = cv2.imread(r'E:\PythonProject\Tongxing\data\test\3.png')
k1 = np.ones((3, 3), np.uint8)

closing1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, k1)
res1=cv2.pyrMeanShiftFiltering(closing1,20,20)
res2=cv2.pyrMeanShiftFiltering(img,20,20)


cv2.imshow("original", img)
cv2.imshow("opening", closing1)
cv2.imshow('mean1',res1)
cv2.imshow('mean2',res2)
cv2.waitKey()
cv2.destroyAllWindows()'''

'''
res1=cv2.pyrMeanShiftFiltering(img,20,20)
res2=cv2.pyrMeanShiftFiltering(res1,20,20)
cv2.imshow('input', img)
cv2.imshow('mean1',res1)
cv2.imshow('mean2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

