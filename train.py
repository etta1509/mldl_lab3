# Import necessary libraries
import os
import torch
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from utils.visualization import *

def train(epoch, model, train_loader, criterion, optimizer): # criterion = loss
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader): #(X,y)
        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute prediction and loss
        outputs = model(inputs) # Contains the prediction of the model of the current batch
        # outputs contains for each row the tuple (class, prob)
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # Update of the loss = contain the total loss of the epoch
        _, predicted = outputs.max(1) # Predicted class
        # outputs.max(0) = the maximum value for each row
        # outputs.max(1) = the predicted class
        total += targets.size(0) # Number of element in the current batch, in total there is the total number of processated batch
        correct += predicted.eq(targets).sum().item() # Sum of correctly predicted element

    train_loss = running_loss / len(train_loader) # Mean loss of the all dataset
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%') # epoch = is the current epoch


if __name__ == "__main__":

    #### LAB 1
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform_1()['train'])
    tiny_imagenet_dataset_test = ImageFolder(root='./dataset/tiny-imagenet-200/test', transform=transform_1()['val'])

    # DataLoader
    dataloader_train, dataloader_test = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_test, 64, True, True)

    # Determine the number of classes and samples
    num_classes = len(tiny_imagenet_dataset_train.classes)
    num_samples = len(tiny_imagenet_dataset_train)

    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    visualization(dataloader_train)


    #### LAB 2 
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform_2())
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/test', transform=transform_2())

    # DataLoader
    train_loader, _ = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_val, 32, True, False)

    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")