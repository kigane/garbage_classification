import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import garbage_dataset
from model import VGG
from utils import Accumulator, DictObj, read_yaml, save_model

_PROJECT = "GarbageClassification"

def correct_num(x: Tensor, y: Tensor) -> int:
    """统计一个批次中预测正确的样本数

    Args:
        x (tensor): size=(batch, class_num) 预测结果
        y (tensor): size=(batch, 1) 真值
    """
    x_label = torch.argmax(x, dim=1)
    return (x_label == y).float().sum()


@torch.no_grad()
def evaluate_accuracy(net: nn.Module, valid_iter: DataLoader, device) -> float:
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


def log_dataset(proj_name) -> None:
    """在wandb上记录数据集"""
    with wandb.init(project=proj_name, job_type="log-data") as run:
        raw_data = wandb.Artifact(
            "garbage-raw", 
            type="dataset",
            description="Raw garbage dataset",
            metadata={
                "classes": "cardboard,glass,metal,paper,plastic,trash",
                "train": 2024,
                "test": 503,
            })
        raw_data.add_dir('dataset')
        run.log_artifact(raw_data)


def log_model(info: DictObj) -> None:
    """在wandb上记录模型"""
    model_artifact = wandb.Artifact(
        name=info.model_name, # 名称
        type=info.model_type, # 类别
        description=info.model_desc,
        metadata=info.model_metadata.to_dict()
    )
    model_artifact.add_file(build_model_path(info.model_name))
    wandb.log_artifact(model_artifact)


def build_model_path(name: str) -> str:
    return 'results/models/' + name + 'pth'


def train(config=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_info = read_yaml('project.yml')

    training_set = garbage_dataset('dataset/train')
    valid_set = garbage_dataset('dataset/valid')
    log_dataset(proj_info.project)

    if config is None:
        wandb.init()
    else:
        wandb.init(project=proj_info.project,
                   group=proj_info.group,
                   job_type=proj_info.job_type,
                   config=config)
    opt = wandb.config

    model = VGG(3, proj_info.model_type).to(device)

    train_iter = DataLoader(
        training_set, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    valid_iter = DataLoader(
        valid_set, batch_size=opt.batch_size, pin_memory=True)

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
            save_model(model, best_acc, build_model_path(proj_info.model_name))

    # after training
    if config is not None:
        log_model(proj_info)


if __name__ == '__main__':
    # single run
    config = read_yaml('options.yml', True)
    train(config)

    # sweep
    # sweep_config = read_yaml('sweep.yml', True)
    # sweep_id = wandb.sweep(sweep_config, project=_PROJECT)
    # wandb.agent(sweep_id, function=train, count=8)
