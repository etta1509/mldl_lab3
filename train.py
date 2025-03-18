# Import necessary libraries
import os
import torch
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from utils.visualization import *


if __name__ == "__main__":

    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform()['train'])
    tiny_imagenet_dataset_test = ImageFolder(root='./dataset/tiny-imagenet-200/test', transform=transform()['val'])

    # DataLoader
    dataloader_train, dataloader_test = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_test)

    # Determine the number of classes and samples
    num_classes = len(tiny_imagenet_dataset_train.classes)
    num_samples = len(tiny_imagenet_dataset_train)

    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    #visualization(dataloader_train)   
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes_sampled = []
    found_classes = 0 # num di classes

    for i, (inputs, classes) in enumerate(dataloader_train):
        img = inputs[i].squeeze()
        label = classes[i]
        if label not in classes_sampled:
            classes_sampled.append(label)
            img_denormalize = denormalize(img)
            found_classes += 1
            plt.subplot(2, 5, i + 1)
            plt.title(f"label:{label}")
            plt.imshow(img_denormalize)
        if found_classes == 10:
            break

    plt.show() 