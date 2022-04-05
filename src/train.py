import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import garbage_dataset
from model import VGG
from utils import Accumulator, get_options, save_model


def init_weights(m):
    """网络权重初始化。作为apply的参数使用。"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def correct_num(x, y):
    """统计一个批次中预测正确的样本数

    Args:
        x (tensor): (batch, class_num) 预测结果
        y (tensor): (batch, 1) 真值
    """
    x_label = torch.argmax(x, dim=1)
    return (x_label == y).float().sum()


def split_dataset(dataset, k):
    """划分数据集。会保持训练集和验证集的类别数据量的分布。

    Args:
        dataset (Dataset): 原始数据集
        k (int): 测试集占原始数据集的1/k

    Returns:
        Tuple(Dataloader, Dataloader): 训练集,测试集的Dataloader
    """
    counter = Counter(dataset.targets)
    train_indices = []
    valid_indices = []
    covered_num = 0
    for i in range(len(counter.keys())):
        class_total = counter.get(i)
        indices = list(range(covered_num, covered_num + class_total))
        split = class_total // k
        # random.shuffle(indices)
        train_indices += indices[split:]
        valid_indices += indices[:split]
        covered_num += class_total
    
    random.shuffle(train_indices)
    random.shuffle(valid_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_iter = DataLoader(
        dataset, batch_size=opt.batch_size, sampler=train_sampler, pin_memory=True)
    valid_iter = DataLoader(
        dataset, batch_size=opt.batch_size, sampler=valid_sampler, pin_memory=True)
    
    return train_iter, valid_iter


@torch.no_grad()
def evaluate_accuracy(net, valid_iter):
    """计算模型在验证集上的准确率

    Args:
        net (nn.Module): 神经网络模型
        valid_iter (Dataloader): 验证集

    Returns:
        float: 模型预测准确率
    """
    net.eval()
    metric = Accumulator(2)
    for X, y in tqdm(valid_iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        metric.add(correct_num(y_hat, y), X.shape[0])
    return metric[0] / metric[1]
        

def train(model, train_iter, valid_iter, opt, train_writer, valid_writer):
    best_acc = 0
    metric = Accumulator(3)
    for epoch in range(opt.n_epochs):
        model.train()
        metric.reset()
        for batch_idx, (X, y) in enumerate(tqdm(train_iter, desc=f'epoch {epoch+1:>3}')):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss.detach().cpu().item() * X.shape[0],
                           correct_num(y_hat.detach().cpu(), y.detach().cpu()),
                           X.shape[0])

        train_writer.add_scalar(
            "train_loss", metric[0] / metric[2], global_step=epoch+1)
        train_writer.add_scalar(
            "train_acc", metric[1] / metric[2], global_step=epoch+1)

        valid_acc = evaluate_accuracy(model, valid_iter)
        valid_writer.add_scalar('valid_acc', valid_acc, global_step=epoch+1)

        if best_acc < valid_acc:
            best_acc = valid_acc
            save_model(model, best_acc, 'VGG13_reduced.pth')


if __name__ == '__main__':
    import os
    for root, dirs, files in os.walk('logs'):
        if len(files) > 0:
            for f in files:
                os.remove(os.path.join(root, f))

    opt = get_options()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_writer = SummaryWriter(f"logs/train")
    valid_writer = SummaryWriter(f"logs/valid")

    dataset = garbage_dataset('dataset')
    train_iter, valid_iter = split_dataset(dataset, 5)

    model = VGG(3, 'VGG13').to(device)
    model.apply(init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train(model, train_iter, valid_iter, opt, train_writer, valid_writer)
