import os
import random
import cv2
import numpy as np
from PIL import Image

path = r"E:\PythonProject\venv\Tongxing\yolov8\runs\classify\predict12\labels"
output_file = open(r"E:\PythonProject\venv\Tongxing\yolov8\runs\classify\predict12\0predict.txt", 'w')
a = ','
num = 0
num_50 = 0
num_75 = 0
num_85 = 0
for filename in os.listdir(path):
    #打开文件
    with open(os.path.join(path,filename), 'r') as f:
        lines = f.readlines()
        out = filename
        for line0 in lines:
            line = line0.strip()
            out += a + line
            con, cla = line.split(' ')
            print(con, cla)
            if int(cla) == 1:
                num += 1
                if float(con) > 0.85:
                    num_85 += 1
                if float(con) > 0.75:
                    num_75 += 1
                if float(con) > 0.50:
                    num_50 += 1
            #print(line[6], end = ' ', file = output_file)

        output_file.write(f"{out}\n")

output_file.write(f"大于85，{num_85}\n大于75，{num_75}\n大于50，{num_50}")
output_file.close()
print(num, num_85, num_75)
