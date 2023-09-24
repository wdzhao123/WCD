import os
import sys
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.optim.lr_scheduler as lr_scheduler

from model import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import datetime
import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.train_data_path, args.test_data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   ])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    model = densenet121(num_classes=args.num_classes).to(device)
    if os.path.exists(args.weights):
        x = torch.load(args.weights, map_location=device)
        model.load_state_dict(x, strict=False)

    # freeze weights
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All but the last fully connected layer are frozen
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    save_path = './weights/{}'.format(datetime.datetime.now())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_acc=0

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / len(val_data_set)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + "/model_{}epoch_{}.pth".format(epoch, round(best_acc, 3)))
        print('best_epoch:{}, best_acc:{}'.format(best_epoch, round(best_acc, 3)))

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   ])}

    # image_path = '../datasets/HRRSD'
    image_path = '../datasets/DOTA_fewshot-10'
    # image_path = '../datasets/HDSSR-fewshot'
    # image_path = '../datasets/NWPU_class'
    # image_path = '../datasets/DIOR'

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    net = densenet121(num_classes=10).to(device)
    model_weight_path = "./weights/2022-11-15 12:29:44.457988/model_791epoch_0.878.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)

    acc = evaluate(model=net, data_loader=validate_loader, device=device, show=True)


    val_accurate = acc / val_num
    print(val_accurate)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--train-data-path', type=str,
                        default="../datasets/DOTA_fewshot-10/train")
    parser.add_argument('--test-data-path', type=str,
                        default="../datasets/DOTA_fewshot-10/test")

    # densenet121
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument('--weights', type=str, default='./pretrain/densenet121-a639ec97.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
    # test()
