import torch
import yaml


class Accumulator:
    """用于累积每个小批量的指标"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DictObj():
    """把字典转换成对象，用点操作符访问"""

    def __init__(self, in_dict):
        for k, v in in_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [DictObj(x) if isinstance(
                    x, dict) else x for x in v])
            else:
                setattr(self, k, DictObj(v) if isinstance(v, dict) else v)


def get_options(filepath='options.yml'):
    """读取yaml文件，获取设置"""
    with open(filepath, 'r') as f:
        yobj = yaml.safe_load(f)
    return DictObj(yobj)


def save_model(model, acc, filename="model.pth"):
    """保存模型"""
    print(f"=> Saving model with acc {acc}")
    torch.save(model.state_dict(), filename)


def load_model(model, filename="model.pth"):
    """加载模型"""
    print("=> Loading model")
    model.load_state_dict(torch.load(filename))


def stat_cuda(msg='GPU memory usage'):
    """打印GPU内存使用情况。
       Pytorch使用缓存内存分配器来加速内存分配。这允许在没有设备同步的情况下快速内存释放。使用memory_allocated()和max_memory_allocated()来监视由张量占据的内存，并使用memory_reserved()max_memory_reserved()来监视缓存分配器管理的存储器总量。

    Args:
        msg (str, optional): 额外信息. Defaults to 'GPU memory usage'.
    """
    print('[{0}] allocated: {1:.2f}M, max allocated: {2:.2f}M, cached: {3}M, max cached: {4}M'.format(
        msg,
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024,
        torch.cuda.max_memory_reserved() / 1024 / 1024
    ))


if __name__ == '__main__':
    from model import VGG
    net = VGG(3)
    save_model(net)

    new_net = VGG(3)
    load_model(new_net)
    x = torch.randn(1, 3, 224, 224)
    print(net(x) == new_net(x))
