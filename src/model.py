import os

import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet101
from tqdm import tqdm

from utils import Accumulator, save_model

VGGs = {
    'VGG11': ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    'VGG13': ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
    'VGG16': ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
}

#----------------------------------------------------------------------------

class VGG(nn.Module):
    
    def __init__(self, in_channels, conv_arch='VGG16'):
        super().__init__()
        self.vgg = self._make_vgg(in_channels, VGGs[conv_arch])
    

    def forward(self, x):
        return self.vgg(x)


    def _make_block(self, num_convs, in_channels, out_channels):
        """搭建一个包含多个卷积层的VGG块. 
        完成一个Stage的任务:通过多层卷积提取特征后下采样, 使特征图宽高减半。

        Args:
            num_convs (int): 包含的卷积层数
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数

        Returns:
            nn.Sequential: VGG块
        """
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)


    def _make_vgg(self, in_channels, conv_arch):
        """构建VGG

        Args:
            in_channels (int): 输入图片的通道数
            conv_arch (list[tuple]): 网络结构

        Returns:
            nn.Module: VGG Net
        """
        conv_blks = []
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self._make_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 6))

#----------------------------------------------------------------------------

class GarbageModel():
    def __init__(self, opt, device):
        self.device = device
        self.opt = opt

        # 构建模型
        # self.net = VGG(3, opt.model_type)
        self.net = self.create_model(opt.model, 6, pretrained=opt.pretrained, freeze=opt.freeze)
        # self.net = resnet101(pretrained=True)
        # self.net.fc = nn.Linear(2048, 6)
        self.net.to(self.device)
        self.net.train()

        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        # 253 for 8, 127 for 16
        self.scheduler = CosineAnnealingLR(self.optimizer, opt.n_epochs * 127, eta_min=1e-7)
        
        # loss = metrics[0]/metrics[2] acc = metrics[1]/metrics[2]
        self.metrics = Accumulator(3)

    def create_model(self, name, num_classes, pretrained=True, freeze=False):
        net = timm.create_model(name, pretrained=pretrained)

        if freeze:
            for param in net.parameters():
                param.requires_grad = False

        if name == "resnet101":
            net.fc = nn.Linear(2048, num_classes)
        elif name == "swin_base_patch4_window7_224":
            net.head = nn.Linear(1024, num_classes)
        elif name == "swin_small_patch4_window7_224":
            net.head = nn.Linear(768, num_classes)

        return net

    def set_input(self, input):
        self.bsize = input[0].shape[0]
        self.X = input[0].to(self.device)
        self.y = input[1].to(self.device)

    def forward(self):
        self.y_hat = self.net(self.X)

    def backward(self):
        self.loss = self.criterion(self.y_hat, self.y)
        self.loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.scheduler.step()
    
    def accumulate_metrics(self):
        with torch.no_grad():
            self.metrics.add(self.loss.float() * self.bsize,
                            correct_num(self.y_hat, self.y),
                            self.bsize)
    
    def reset_metrics_net(self):
        self.metrics.reset()
        self.net.train()
    
    def calc_acc(self):
        return self.metrics[1] / self.metrics[2]

    def calc_loss(self):
        return self.metrics[0] / self.metrics[2]
    
    def save(self, best_acc):
        save_model(self.net, best_acc, 
                   os.path.join(self.opt.model_path, self.opt.model + '.pth'))

#----------------------------------------------------------------------------

@torch.no_grad()
def correct_num(x, y):
    """统计一个批次中预测正确的样本数

    Args:
        x (tensor): size=(batch, class_num) 预测结果
        y (tensor): size=(batch, 1) 真值
    """
    x_label = torch.argmax(x, dim=1)
    return (x_label == y).float().sum()


@torch.no_grad()
def evaluate_accuracy(net, valid_iter, device):
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


if __name__ == '__main__':
    net = resnet101(pretrained=True)
    net.fc = nn.Linear(2048, 6)
    # X = torch.randn(size=(1, 1, 224, 224))
    # for blk in net.vgg:
    #     X = blk(X)
    #     print(blk.__class__.__name__, 'output shape:\t', X.shape)
    for m in net.named_modules():
        print(m)
    # from utils import calc_net_params
    # calc_net_params('VGG16', net)
