'''
import os
from keras.utils.image_utils import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen1 = ImageDataGenerator(
    horizontal_flip=True,  # 水平翻转
    vertical_flip=True,  # 垂直翻转
    rotation_range=0,  # 旋转角度范围，默认为0，表示不进行旋转
    width_shift_range=0,  # 水平平移范围，默认为0，表示不进行水平平移
    height_shift_range=0,  # 垂直平移范围，默认为0，表示不进行垂直平移
    shear_range=0,  # 错切变换范围，默认为0，表示不进行错切变换
    zoom_range=0,  # 缩放范围，默认为0，表示不进行缩放
    fill_mode='nearest'  # 填充方式
)

# 图片文件夹路径
folder_path = r'E:\PythonProject\venv\Tongxing\data0\train\0'

# 加载文件夹中的全部图片
images = []
a=0
for filename in os.listdir(folder_path):
    img = load_img(os.path.join(folder_path, filename))
    x = img_to_array(img)
    images.append(x)
    # 将图片列表转换为numpy数组
    x = np.array(images[a])
    # 扩展维度
    x = np.expand_dims(x, 0)
    print(x.shape)
    # 打印结果：
    # (1, 414, 500, 3)
    a=a+1
    # 打印结果：(num_images, height, width, channels)
    # 生成2张额外图片
    i = 0
    for batch in datagen1.flow(x, batch_size=1,save_to_dir=r'E:\PythonProject\venv\Tongxing\data0\train\0', save_prefix=filename+str(i), save_format='jpg'):
        i += 1
        if i == 1:
            break
    print('finished!')
'''

import os
import random
import cv2
import numpy as np
from PIL import Image

def deal(image):
    w = image.shape[1]
    h = image.shape[0]
    '''
    #亮度调整
    percetage = random.uniform(0.8, 1.5)
    image_copy = image.copy()
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    image = image_copy.copy()
    print('0')

    #图像缩放
    if w > 1000 or h > 1000:
        scale = random.uniform(0.8, 1)
    elif w < 500 or h < 500:
        scale = random.uniform(1, 1.2)
    else:
        scale = 1
    print('0')'''

    #角度变化
    (cX, cY) = (w // 2, h // 2)
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    dst = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

    #翻转变化
    if(random.randint(0,1)):
        dst = np.fliplr(dst)
    return dst

def stack(img1, img2, img3, img4):
    # 大图宽高，及拼接中心点
    w1 = max(img1.shape[0], img3.shape[0])
    w2 = max(img2.shape[0], img4.shape[0])
    h1 = max(img1.shape[1], img2.shape[1])
    h2 = max(img3.shape[1], img4.shape[1])

    #创建大图
    dst = np.ones((h1+h2, w1+w2, 3), dtype=np.uint8) * 255

    dst[h1-img1.shape[0]:h1, w1-img1.shape[1]:w1, :] = img1[:, :, :]
    dst[h1-img2.shape[0]:h1, w1:w1+img2.shape[1], :] = img2[:, :, :]
    dst[h1:h1+img3.shape[0], w1-img3.shape[1]:w1, :] = img3[:, :, :]
    dst[h1:h1+img4.shape[0], w1:w1+img4.shape[1], :] = img4[:, :, :]
    #trans_image[:h1, :w1, :] = transformed_image[:, :, :]
    #dst = np.hstack((org_image[:, :w0, :], trans_image[:, :w1, :]))

    return dst
'''
path = 'E:/PythonProject/venv/Tongxing/data/river_filter/1'
path0 = 'E:/PythonProject/venv/Tongxing/data/river_filter/train/1/'
for filename in os.listdir(path):
    #原图
    img1 = cv2.imread(path + '/' + filename)
    height, width = img1.shape[:2]
    #print(img1.shape)

    # 在原图上和目标图像上各选三个点
    m = np.random.randint(10,width/4)
    n = np.random.randint(10,height/4)
    mat_src = np.float32([[0, 15], [0, height - 20], [width - 1, 0]])
    mat_dst = np.float32([[0, 15], [m, height - n], [width - m, n]])
    # 获得变换矩阵
    mat_trans = cv2.getAffineTransform(mat_src, mat_dst)
    # 进行仿射变换
    img2 = cv2.warpAffine(img1, mat_trans, (width, height), borderValue=(0, 0, 0))

    # 旋转变化
    img3 = deal(img1)
    img4 = deal(img2)
    w = img1.shape[0]
    h = img1.shape[1]

    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    img3 = cv2.resize(img3, (w, h))
    img4 = cv2.resize(img4, (w, h))
    #dst = stack(img1, img2, img3, img4)

    #cv2.namedWindow("imgs", cv2.WINDOW_NORMAL)
    imgs = np.hstack([img1, img2, img3, img4])
    #cv2.imshow("imgs", dst)
    cv2.imwrite(path0 + filename, imgs)
    print(imgs.shape)
    #cv2.waitKey(0)
'''
def rotate(image):
    h, w = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    dst = cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))
    return dst

path = r'E:/PythonProject/venv/Tongxing/yolov8/datasets/river/test/2/'
path0 = r'E:\PythonProject\venv\Tongxing\yolov8\datasets\river\train\0'
for filename in os.listdir(path):
    #原图
    img1 = cv2.imread(os.path.join(path, filename))
    #img1 = Image.open(os.path.join(path, filename)).convert("RGB")
    height, width = img1.shape[:2]
    center = (width / 2, height / 2)
    #旋转
    img2 = rotate(img1)
    #img3 = rotate(img2)
    #img4 = rotate(img3)
    #镜像
    #img2 = cv2.flip(img1, 1)
    img3 = cv2.flip(img1, 0)
    img4 = cv2.flip(cv2.flip(img2, 0), 1)

    '''
    cv2.imwrite(os.path.join(path0, '0' + filename), img1)
    cv2.imwrite(os.path.join(path0, '1' + filename), img2)
    cv2.imwrite(os.path.join(path0, '2' + filename), img3)
    cv2.imwrite(os.path.join(path0, '3' + filename), img4)
    '''
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))
    img3 = cv2.resize(img3, (width, height))
    img4 = cv2.resize(img4, (width, height))

    img12 = np.concatenate([img1, img2], axis=1)
    img34 = np.concatenate([img3, img4], axis=1)

    img0 = np.concatenate([img12, img34], axis=0)
    print(img1.shape, img0.shape)
    cv2.imwrite(os.path.join(path0, '4' + filename), img0)