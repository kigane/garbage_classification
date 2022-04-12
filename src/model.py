from turtle import forward

import torch
import torch.nn as nn

VGGs = {
    'VGG11': ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    'VGG13': ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
    'VGG16': ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
}


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


if __name__ == '__main__':
    net = VGG(1)
    X = torch.randn(size=(1, 1, 224, 224))
    for blk in net.vgg:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)
