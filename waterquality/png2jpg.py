import os
from PIL import Image

if __name__ == '__main__':
    onefileDir = r"E:/PythonProject/venv/Tongxing/yolov8/datasets/river_4/images"  # 源图片文件夹路径
    '''
    for oneDir in os.listdir(filePath):
        onefileDir = os.path.join(filePath, oneDir)
        print(onefileDir)
        for filename in os.listdir(onefileDir):
            name = filename.split('.')[0]
            end = filename.split('.')[1]
            print(end)
            if (end == 'png'):
                image = Image.open(os.path.join(onefileDir, filename))
                im = image.convert('RGB')
                im.save(os.path.join(onefileDir, name) + '.jpg', quality=95)
                print(filename)
                os.remove(os.path.join(onefileDir, filename))'''
    for filename in os.listdir(onefileDir):
        name = filename.split('.')[0]
        end = filename.split('.')[1]
        print(end)
        if (end == 'png'):
            image = Image.open(os.path.join(onefileDir, filename))
            im = image.convert('RGB')
            im.save(os.path.join(onefileDir, name) + '.jpg', quality=95)
            print(filename)
            os.remove(os.path.join(onefileDir, filename))



        # 删除原文件夹（这个时候文件夹应该是已经空了的）
        #os.removedirs(onefileDir)