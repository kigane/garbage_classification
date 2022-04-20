import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.utils import make_grid

from dataset import garbage_dataset
from model import VGG
from utils import load_model


def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(2048, 6)
    load_model(model, 'results/models/Resnet101.pth')
    model.eval()

    valid_set = garbage_dataset('dataset/valid')
    n = 16
    imgs = []
    for i in range(n):
        imgs.append(valid_set[i][0])
    t = torch.stack(imgs)

    print(t.shape)
    # show_tensor_images(t, num_images=16, nrow=4)
    fig, axes = plt.subplots(4, 4)
    axes = axes.reshape(-1)
    for i, (img, ax) in enumerate(zip(imgs, axes)):
        img = (img + 1)/2
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(str(i))
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.5)  # 调整子图间距
    plt.show()
    # valid_iter = DataLoader(valid_set, batch_size=8)
    # plt.imshow(valid_set[0][0].permute(1, 2, 0))
    # plt.show()

    
    
