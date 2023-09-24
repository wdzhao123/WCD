import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

def read_split_data(train_root: str, test_root:str, val_rate: float = 0.2):
    random.seed(0)  # Random results are guaranteed to be reproducible
    assert os.path.exists(train_root), "dataset root: {} does not exist.".format(train_root)
    assert os.path.exists(test_root), "dataset root: {} does not exist.".format(test_root)

    classes = [cla for cla in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp"]

    for cla in classes:
        cla_path = os.path.join(train_root, cla)

        images = [os.path.join(train_root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    for cla in classes:
        cla_path = os.path.join(test_root, cla)

        images = [os.path.join(test_root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:

        plt.bar(range(len(classes)), every_class_num, align='center')

        plt.xticks(range(len(classes)), classes)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')

        plt.ylabel('number of images')

        plt.title('class distribution')
        # plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data_raw(root: str, val_rate: float = 0.2):
    random.seed(0)  # Random results are guaranteed to be reproducible
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp"]

    for cla in classes:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:

        plt.bar(range(len(classes)), every_class_num, align='center')

        plt.xticks(range(len(classes)), classes)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')

        plt.ylabel('number of images')

        plt.title('class distribution')
        # plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, show=False):
    model.eval()


    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader)
    pre_dic = []
    label_dic = []
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
        for pre in pred:
            pre_dic.append(pre.item())
        for lab in labels:
            label_dic.append(lab)
    if show == True:
        pre_dic = np.array(pre_dic)
        label_dic = np.array(label_dic)
        lable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cm = confusion_matrix(label_dic, pre_dic, labels=lable)

        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        precision = np.sum(precision) * 0.1
        recall = np.sum(recall) * 0.1
        print("precision:{}".format(precision))
        print("recall:{}".format(recall))

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(lable))
        plt.xticks(tick_marks, lable, rotation=45, fontsize=5)
        plt.yticks(tick_marks, lable, fontsize=7)

        fmt = '.1f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=5,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Prediction')
        plt.xlabel('Label')
        # plt.savefig('./figures/Ours.pdf', dpi=200, bbox_inches='tight')

        # plt.show()

    return sum_num.item()
