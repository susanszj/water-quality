import os
from torchvision import transforms, datasets
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import repvgg
from repvgg import RepVGG, RepVGGBlock
from PIL import Image
import cv2
import numpy as np
import scipy.signal as ss
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
    INPUT_DICT = 'best2.pth'

    #model = repvgg.repvgg_model_convert(torch.load(INPUT_DICT, map_location=torch.device('cuda:0')))
    model = torch.load(INPUT_DICT, map_location=torch.device('cuda:0'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    '''
    path0 = r"C:/Users/25005/Desktop/1.png"
    path1= r"C:/Users/25005/Desktop/2.png"
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
    image = Image.open(path1).convert('RGB')

    # image = image.unsqueeze(0)
    tf = transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = tf(image).unsqueeze(0)
    outputs = model(image.to(device))

    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu()

    # 打印图片信息
    #print(f"图片名称: {filename}")
    print(f"分类结果: {predicted.item()}")

    # 获取置信度
    confidence = torch.max(outputs).item()
    print(f"置信度: {outputs}")
    print()

    '''
    a=0
    num_50 = 0
    num_70 = 0
    num_85 = 0
    path = r"E:\PythonProject\venv\Tongxing\yolov8\datasets\predict\0"
    output_file = open(r'E:\PythonProject\venv\Tongxing\yolov8\datasets\predict\0predict.txt', 'w')
    for filename in os.listdir(path):
        image =  Image.open(os.path.join(path, filename)).convert('RGB')
        
        #image = image.unsqueeze(0)
        tf = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image = tf(image).unsqueeze(0)
        outputs = model(image.to(device))

        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()

        # 打印图片信息
        print(f"图片名称: {filename}")
        print(f"分类结果: {predicted.item()}")

        # 获取置信度
        confidence = torch.max(outputs).item()
        print(f"置信度: {confidence}")
        print()

        if (predicted.item() == 1):
            a = a + 1
            if confidence > 0.50:
                num_50 += 1
            if confidence > 0.70:
                num_70 += 1
            if confidence > 0.85:
                num_85 += 1


        # 将打印结果写入文件
        output_file.write(f"{filename}  ,  {predicted.item()}  ,  {confidence}\n")
    print(f"污染数为：{a}")
    output_file.write(f"> 50:{num_50}\n> 70:{num_70}\n> 85:{num_85}\n")
    # 关闭输出文件
    output_file.close()
