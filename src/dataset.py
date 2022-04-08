from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def preprocess_transform(resize=224):
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ToTensor()
    ])


def garbage_dataset(root):
    transform = preprocess_transform()
    dataset = ImageFolder(root, transform=transform)
    dataset.idx_to_class = {v:k for k,v in dataset.class_to_idx.items()}
    return dataset


if __name__ == '__main__':
    dataset = garbage_dataset('dataset')
