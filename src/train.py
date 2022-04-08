import random
from collections import Counter
from typing import Any, Tuple

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from dataset import garbage_dataset
from model import VGG
from utils import Accumulator, read_yaml, save_model

_PROJECT = "GarbageClassification"

def init_weights(m: nn.Module) -> None:
    """网络权重初始化。作为apply的参数使用。"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def correct_num(x: Tensor, y: Tensor) -> int:
    """统计一个批次中预测正确的样本数

    Args:
        x (tensor): size=(batch, class_num) 预测结果
        y (tensor): size=(batch, 1) 真值
    """
    x_label = torch.argmax(x, dim=1)
    return (x_label == y).float().sum()


def split_dataset(dataset: Dataset, k: int) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
    """划分数据集。会保持训练集和验证集的类别数据量的分布。

    Args:
        dataset (Dataset): 原始数据集
        k (int): 测试集占原始数据集的1/k

    Returns:
        Tuple(Sampler, Sampler): 训练集,测试集的Sampler
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
    
    return train_sampler, valid_sampler


@torch.no_grad()
def evaluate_accuracy(net: nn.Module, valid_iter: DataLoader, device: Any) -> float:
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


def log_dataset(size: int) -> None:
    """在wandb上记录数据集"""
    with wandb.init(project=_PROJECT, job_type="log-data") as run:
        raw_data = wandb.Artifact(
            "garbage-raw", type="dataset",
            description="Raw garbage dataset",
            metadata={
                "classes": "cardboard,glass,metal,paper,plastic,trash",
                "sizes": size
            })
        raw_data.add_dir('dataset')
        run.log_artifact(raw_data)


def log_model(name: str) -> None:
    """在wandb上记录模型"""
    model_artifact = wandb.Artifact(
        name='name',
        type='trained_model',
        description='VGG13 with linear layer reduced',
        metadata={'use_BN': True, 'activation': 'relu'}
    )
    model_artifact.add_file('results/models/' + name + 'pth')
    wandb.log_artifact(model_artifact)


def train(config=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = garbage_dataset('dataset')
    log_dataset(len(dataset))

    if config is None:
        wandb.init()
    else:
        wandb.init(project=_PROJECT, group='VGG',
               job_type='train VGG', config=config)
    opt = wandb.config

    model = VGG(3, 'VGG13').to(device)
    model.apply(init_weights)

    train_sampler, valid_sampler = split_dataset(dataset, 5)
    train_iter = DataLoader(
        dataset, batch_size=opt.batch_size, sampler=train_sampler, pin_memory=True)
    valid_iter = DataLoader(
        dataset, batch_size=opt.batch_size, sampler=valid_sampler, pin_memory=True)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

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
                metric.add(loss.float() * X.shape[0],
                           correct_num(y_hat, y),
                           X.shape[0])

        # after train loop
        wandb.log({'train/loss': metric[0] / metric[2]}, step=epoch+1)
        wandb.log({'train/acc': metric[1] / metric[2]}, step=epoch+1)

        valid_acc = evaluate_accuracy(model, valid_iter, device)
        wandb.log({'valid/acc': valid_acc}, step=epoch+1)

        if best_acc < valid_acc:
            best_acc = valid_acc
            wandb.summary['best_acc'] = best_acc
            save_model(model, best_acc, 'results/models/VGG13_reduced.pth')

    # after training
    if config is not None:
        log_model('VGG13_reduced_.pth')


if __name__ == '__main__':
    # config = read_yaml('options.yml', True)
    # train(config)
    sweep_config = read_yaml('sweep.yml', True)
    sweep_id = wandb.sweep(sweep_config, project=_PROJECT)
    wandb.agent(sweep_id, function=train, count=8)
