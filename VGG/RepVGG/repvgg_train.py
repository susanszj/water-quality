import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from repvgg import create_RepVGG_A0, create_RepVGG_B0
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_and_val(epochs, model, train_loader, val_loader, criterion):
    torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0
    learning_rate = 0.0001
    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        optimizer = optim.Adam(net.parameters(), lr = learning_rate)  # 设置优化器和学习率 betas=(0.9, 0.999),
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:
                # training phase

                #                 images, labels = data
                #             optimizer.zero_grad()
                #             logits = net(images.to(device))
                #             loss = loss_function(logits, labels.to(device))
                #             loss.backward()
                #             optimizer.step()

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                # forward
                output = model(image)
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]

                # backward
                loss.backward()
                optimizer.step()  # update weight

                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()
                pbar.update(1)

        model.eval()
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]

                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculatio mean for each batch
            train_loss.append(running_loss / len(train_dataset))
            val_loss.append(val_losses / len(val_dataset))

            train_acc.append(training_acc / len(train_dataset))
            val_acc.append(validation_acc / len(val_dataset))

            torch.save(model, "last2.pth")
            if best_acc < (validation_acc / len(val_dataset)):
                best_acc = validation_acc / len(val_dataset)
                torch.save(model, "best2.pth")
                print('\nreplace!')
                print(best_acc)

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Acc: {:.3f}..".format(training_acc / len(train_dataset)),
                  "Val Acc: {:.3f}..".format(validation_acc / len(val_dataset)),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_dataset)),
                  "Val Loss: {:.3f}..".format(val_losses / len(val_dataset)),
                  "Time: {:.2f}s".format((time.time() - since)))

        classes = ('n0', 'n1')

        class_correct = [0.] * 10
        class_total = [0.] * 10
        y_test, y_pred = [], []
        X_test = []

        with torch.no_grad():
            for images, labels in val_loader:
                X_test.extend([_ for _ in images])
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                c = (predicted == labels).squeeze()
                for i, label in enumerate(labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                y_pred.extend(predicted.numpy())
                y_test.extend(labels.cpu().numpy())

        for i in range(2):
            print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")

    history = {'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    BATCH_SIZE = 32

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder("E:/PythonProject/venv/Tongxing/yolov8/datasets/river/train/", transform=data_transform["train"])  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)  # 加载数据

    val_dataset = datasets.ImageFolder("E:/PythonProject/venv/Tongxing/yolov8/datasets/river/test/", transform=data_transform["val"])  # 测试集数据
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=2)  # 加载数据

    #print(train_dataset)

    net = create_RepVGG_B0()
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    epoch = 100

    history = train_and_val(epoch, net, train_loader, val_loader, loss_function)

    def plot_loss(x, history):
        plt.plot(x, history['val_loss'], label='val', marker='o')
        plt.plot(x, history['train_loss'], label='train', marker='o')
        plt.title('Loss per epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()


    def plot_acc(x, history):
        plt.plot(x, history['train_acc'], label='train_acc', marker='x')
        plt.plot(x, history['val_acc'], label='val_acc', marker='x')
        plt.title('Score per epoch')
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()

    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)

    classes = ('n0', 'n1')

    class_correct = [0.] * 10
    class_total = [0.] * 10
    y_test, y_pred = [], []
    X_test = []

    with torch.no_grad():
        for images, labels in val_loader:
            X_test.extend([_ for _ in images])
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            c = (predicted == labels).squeeze()
            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1
            y_pred.extend(predicted.numpy())
            y_test.extend(labels.cpu().numpy())

    for i in range(2):
        print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")

    from sklearn.metrics import confusion_matrix, classification_report

    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes)
    print("Accuracy is :", ac)
    print(cr)