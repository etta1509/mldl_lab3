import matplotlib.pyplot as plt
from dataset.denormalize import *

# Visualize one example for each class for 10 classes
def visualization(dataloader_train):
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
        if found_classes == 9:
            break

    plt.show()