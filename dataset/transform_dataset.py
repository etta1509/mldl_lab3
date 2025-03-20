import torchvision.transforms as transforms
import shutil
import os

# Define transformations for the dataset for the lab 1
def transform_1():
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return transform

# For the lab 2
def transform_2():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform

def modify_dataset():
    with open('./dataset/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'./dataset/tiny-imagenet-200/val/{cls}', exist_ok=True)
            shutil.copyfile(f'./dataset/tiny-imagenet-200/val/images/{fn}', f'./dataset/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('./dataset/tiny-imagenet-200/val/images')