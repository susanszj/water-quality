import os, random, shutil
import pandas as pd
'''
# 设置源文件夹路径
source_folder = r'./train_image'
# 设置目标文件夹路径
destination_folder = r'./'

# 读取Excel文件
excel_file = r'./train_label.csv'
df = pd.read_csv(excel_file)

# 获取第二列属性值（假设是'属性'列）
attribute_column = df['label']
# print(attribute_column)

# 创建目标文件夹
for i in range(1, 5):
    folder_name = os.path.join(destination_folder, f'{i-1}')
    os.makedirs(folder_name, exist_ok=True)

# 遍历源文件夹中的图片
for filename in os.listdir(source_folder):
    # 获取文件的属性值
    file_attribute = attribute_column[df['image_name'] == filename].values[0]
    # print(file_attribute)

    # 根据属性值移动文件到相应的目标文件夹
    if file_attribute == 0:
        destination = os.path.join(destination_folder, '0')
    elif file_attribute == 1:
        destination = os.path.join(destination_folder, '1')
    elif file_attribute == 2:
        destination = os.path.join(destination_folder, '2')
    else:
        destination = os.path.join(destination_folder, '3')

    shutil.move(os.path.join(source_folder, filename), destination)

'''
#划分train和val
def moveFile(fileDir, tarDir, perc):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)

    picknumber = int(filenumber * perc)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片

    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    '''
    rate_test = 0.1
    picknumber_test = int(filenumber * rate_test)
    sample_test = random.sample(pathDir, picknumber_test)

    for name in sample_test:
        if name not in os.listdir(tarDir_val):
            shutil.move(fileDir + name, tarDir_test + name)

    for name in os.listdir(fileDir):
        shutil.move(fileDir + name, tarDir_train + name)'''


if __name__ == '__main__':
    filePath = "E:/PythonProject/venv/Tongxing/data/river_filter/"  # 源图片文件夹路径

    train_fileDir = "E:/PythonProject/venv/Tongxing/data/river_filter/train/"
    val_fileDir = "E:/PythonProject/venv/Tongxing/data/river_filter/val/"
    test_fileDir = "E:/PythonProject/venv/Tongxing/data/river_filter/test/"

    for oneDir in os.listdir(filePath):
        onefileDir = filePath + oneDir + "/"
        onetarDir_train = train_fileDir + oneDir + "/"  # A的二级目录
        onetarDir_val = val_fileDir + oneDir + "/"  # B的二级目录
        onetarDir_test = test_fileDir + oneDir + "/"
        print(onefileDir)
        print(onetarDir_train)
        print(onetarDir_val)
        print(onetarDir_test, end="\n\n")

        # 判断文件夹是否存在，不存在则创建
        if not os.path.exists(onetarDir_train):
            os.makedirs(onetarDir_train)
        if not os.path.exists(onetarDir_val):
            os.makedirs(onetarDir_val)
        if not os.path.exists(onetarDir_test):
            os.makedirs(onetarDir_test)

        moveFile(onefileDir, onetarDir_val, 0.2)
        moveFile(onefileDir, onetarDir_test, 0.125)


        # 删除原文件夹（这个时候文件夹应该是已经空了的）
        #os.removedirs(onefileDir)

