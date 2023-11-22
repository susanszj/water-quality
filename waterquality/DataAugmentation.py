#!usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from imgaug import augmenters as iaa
import os


class MyAugMethod():

    def __init__(self):
        self.seq = iaa.Sequential()
        self.imglist_name = []
        self.imglist = []

    # 遍历输入文件夹，返回所有图片名称
    def show_path_file(self, inputpath, all_files_name, all_files):
        # 首先遍历当前目录所有文件及文件夹
        file_list = os.listdir(inputpath)
        # 保存图片文件的目录
        last_path = inputpath
        # 准备循环判断每个元素是否是文件夹还是文件，
        # 是文件的话，把名称传入list，是文件夹的话，递归
        for filename in file_list:
            # 利用os.path.join()方法取得路径全名，并存入cur_path变量
            # 否则每次只能遍历一层目录
            cur_path = os.path.join(inputpath, filename)
            # 判断是否是文件夹
            if os.path.isdir(cur_path):
                last_path = cur_path
                self.show_path_file(cur_path, all_files_name, all_files)
            else:
                filename = os.path.join(last_path, filename)
                all_files_name.append(filename)
                all_files.append(cv2.imread(filename))

                # 定义增强的方法

    def aug_method(self):
        # 给指定的方法设置对应比例
        # 如Sometimes(0.5, GaussianBlur(0.3))表示每两张图片做一次模糊处理
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # 定义一组变换方法.
        self.seq = iaa.Sequential([
            # 选择0到5种方法做变换
            iaa.SomeOf((0, 5),
                       [
                           # 使用不同的模糊方法来对图像进行模糊处理
                           # 高斯滤波
                           # 均值滤波
                           # 中值滤波
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # 对图像进行锐化处理，alpha表示锐化程度
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                           # 与sharpen锐化效果类似，但是浮雕效果
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # 添加高斯噪声
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255)
                           ),

                           # 每个像素增加（-10,10）之间的像素值
                           iaa.Add((-10, 10), per_channel=0.5),

                           # 将-40到40之间的随机值添加到图像中，每个值按像素采样
                           iaa.AddElementwise((-40, 40)),

                           # 改变图像亮度（原值的50-150%）
                           iaa.Multiply((0.5, 1.5)),

                           # 将每个像素乘以0.5到1.5之间的随机值.
                           iaa.MultiplyElementwise((0.5, 1.5)),

                           # 增强或弱化图像的对比度.
                           iaa.ContrastNormalization((0.5, 2.0)),
                       ],
                       # 按随机顺序进行上述所有扩充
                       random_order=True
                       )

        ], random_order=True)

        # 增强函数

    def aug_data(self, inputpath, times):
        # 获得输入文件夹中的文件列表
        self.show_path_file(inputpath, self.imglist_name, self.imglist)
        # 实例化增强方法
        self.aug_method()
        # 对文件夹中的图片进行增强操作，循环times次
        for count in range(times):
            print("aug data for {} times ".format(count))
            images_aug = self.seq.augment_images(self.imglist)
            for index in range(len(images_aug)):
                filename = self.imglist_name[index].split(".jpg", 1)[0]
                filename = filename + "_" + str(count) + ".jpg"
                # 保存图片
                cv2.imwrite(filename, images_aug[index])
                # print('image of count%s index%s has been writen'%(count,index))


if __name__ == "__main__":
    # 图片文件相关路径
    inputpath = 'E:/PythonProject/venv/Tongxing/data/data3/train/1/'
    times = 2
    test = MyAugMethod()
    test.aug_data(inputpath, times)