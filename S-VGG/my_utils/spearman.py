import scipy.stats as stats
import torch
from audtorch.metrics.functional import pearsonr

def func(x, y):
    batch_size = x.shape[0]
    n = x.shape[1]
    x_tuple = torch.split(x, 1, dim=0)
    y_tuple = torch.split(y, 1, dim=0)
    sum = 0
    for i in range(batch_size):
        # a = x_tuple[i].cpu().detach().numpy().squeeze()
        # b = y_tuple[i].cpu().detach().numpy().squeeze()
        # a, _ = stats.spearmanr(a, b)
        a = x_tuple[i].detach().squeeze()
        b = y_tuple[i].detach().squeeze()
        a = pearsonr(a, b)

        sum += a
    return sum / batch_size

def count_spear(img, net1, net2, net3, device):

    f1 = net1(img.to(device))
    f2 = net2(img.to(device))
    f3 = net3(img.to(device))

    t12 = func(f1, f2)
    t13 = func(f1, f3)
    t23 = func(f2, f3)

    return f1, f3


