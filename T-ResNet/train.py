import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet50
import datetime

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   ])}

    image_path = '../datasets/DOTA_fewshot'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet50(num_classes=15)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "checkpoints/1.1-不加载预训练模型/43epoch_0.936.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    x = torch.load(model_weight_path, map_location=device)
    # del x['fc.weight']
    # del x['fc.bias']
    net.load_state_dict(x, strict=False)
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    epochs = 1000
    best_acc = 0.0
    now = datetime.datetime.now()
    save_path = './model/' + str(now)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch
            torch.save(net.state_dict(), save_path + '/{}epoch_{}.pth'.format(best_epoch, round(best_acc, 3)))
        print('best_epoch:{}, best_acc:{}'.format(best_epoch, round(best_acc, 3)))

    print('Finished Training')

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   ])}

    image_path = '/home/lxz/遥感分类paper2/datasets/DOTA_fewshot-10'
    # image_path = '/home/lxz/遥感分类paper2/datasets/HRRSD'
    # image_path = '/home/lxz/遥感分类paper2/datasets/DIOR'
    # image_path = '/home/lxz/遥感分类paper2/datasets/NWPU_class'

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    net = resnet50(num_classes=10).to(device)
    model_weight_path = "./572epoch_0.882.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)

    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    pre_dic = []
    label_dic = []
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs= net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            for pre in predict_y:
                pre_dic.append(pre.item())
            for lab in val_labels:
                label_dic.append(lab.item())

    val_accurate = acc / val_num
    print(val_accurate)

    pre_dic = np.array(pre_dic)
    label_dic = np.array(label_dic)
    lable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # lable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    cm = confusion_matrix(label_dic, pre_dic, labels=lable)

    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.sum(precision) * 0.1
    recall = np.sum(recall) * 0.1
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))

    # 如果希望混淆矩阵行为100
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 如果希望混淆矩阵列为100,用到转置矩阵
    cm = np.transpose(cm)
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :] * 100

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


if __name__ == '__main__':
    # main()
    test()