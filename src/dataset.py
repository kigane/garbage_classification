from collections import Counter

import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def preprocess_transform(resolution=224):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])


def garbage_dataset(root, is_train=True, resolution=224):
    if is_train:
        transform = preprocess_transform()
    else:
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], 
                                 [0.5, 0.5, 0.5]),
        ])
    dataset = ImageFolder(root, transform=transform)
    dataset.idx_to_class = {v:k for k,v in dataset.class_to_idx.items()}
    return dataset


def split_dataset(dataset, k):
    """划分数据集。会保持训练集和验证集的类别数据量的分布。

    Args:
        dataset (Dataset): 原始数据集
        k (int): 测试集占原始数据集的1/k

    Returns:
        Tuple(Subset, Subset): 训练集,测试集
    """
    counter = Counter(dataset.targets)
    train_indices = []
    valid_indices = []
    covered_num = 0
    for i in range(len(counter.keys())):
        class_total = counter.get(i)
        indices = list(range(covered_num, covered_num + class_total))
        split = class_total // k
        train_indices += indices[split:]
        valid_indices += indices[:split]
        covered_num += class_total
    return Subset(dataset, train_indices), Subset(dataset, valid_indices)


if __name__ == '__main__':
    from utils import get_stat
    train = garbage_dataset('dataset/train', is_train=False)
    # print(len(train))
    # valid = garbage_dataset('dataset/valid')
    # print(len(valid))
    # print(get_stat(train))
    # print(get_stat(valid))
    img,  _ = train[16]
    c = transforms.ColorJitter(brightness=0.3, saturation=0.3)
    # img = c(img)
    fig, axes = plt.subplots(1, 2)
    axes = axes.reshape(-1)
    axes[0].imshow(img.permute(1, 2, 0))
    axes[1].imshow(c(img).permute(1, 2, 0))
    plt.show()
