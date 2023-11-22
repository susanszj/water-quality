import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from train import ResNet, BasicBlock, Bottleneck
from PIL import Image

if __name__=="__main__":
    INPUT_DICT = 'best_resnet34_0.86.pth'

    model = torch.load(INPUT_DICT, map_location=torch.device('cuda:0'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    BATCH_SIZE = 16

    '''data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder("./train/", transform=data_transform["train"])  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)  # 加载数据

    val_dataset = datasets.ImageFolder("./val/", transform=data_transform["val"])  # 测试集数据
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=2)  # 加载数据

    classes = ('n0', 'n1', 'n2', 'n3')

    class_correct = [0.] * 4
    class_total = [0.] * 4
    y_test, y_pred = [], []
    X_test = []

    with torch.no_grad():
        for images, labels in val_loader:
            X_test.extend([_ for _ in images])
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            c = (predicted == labels).squeeze()
            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1
            y_pred.extend(predicted.numpy())
            y_test.extend(labels.cpu().numpy())

    for i in range(4):
        print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")

    from sklearn.metrics import confusion_matrix, classification_report

    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes)
    print("Accuracy is :", ac)
    print(cr)'''
    image =  Image.open("E:/PythonProject/venv/Tongxing/data/data3/train/0/AzdqV.jpg").convert('RGB')
    #image = image.unsqueeze(0)
    tf = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = tf(image).unsqueeze(0)
    outputs = model(image.to(device))
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu()
    print(predicted)

