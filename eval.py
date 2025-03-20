# Import necessary libraries
import os
import torch
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from models.custom_net import *
from train import train, train_loader # Attenzione può essere sbagliato!

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs) # Predicted
            loss = criterion(outputs, targets) # Computation of the loss

            # As computed in the train part
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

if __name__ == "__main__":

    #### LAB 2 
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform_2())
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/test', transform=transform_2())

    # DataLoader
    _, val_loader = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_val, 32, True, False)

    print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

    # Train + Eval of the model 
    model = CustomNet().cuda() # The creation of the model
    criterion = nn.CrossEntropyLoss() # Loss
    # change the parameter lr and momentum (?)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # Use Stocastic Gradient Descent to minize the loss (optimization of the parameters)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        # We train the model
        train(epoch, model, train_loader, criterion, optimizer) # Give us input the current epoch
        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion) # Compute the accuracy on the validation set
        print(f"val_accuracy: {val_accuracy}")
        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)
        print(f"best_acc: {best_acc}")

    print(f'Best validation accuracy: {best_acc:.2f}%')