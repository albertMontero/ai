import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_hist(history):
    x_arr = np.arange(len(history['train_loss'])) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, history['train_loss'], '-o', label='train_loss')
    ax.plot(x_arr, history['val_loss'], '-->', label='val_loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=12)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, history['train_accuracy'], '-o', label='train_acc')
    ax.plot(x_arr, history['val_accuracy'], '-->', label='val_acc')
    ax.legend(fontsize=12)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)

    plt.show()

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model_predictions(model, data_transforms, class_names, img_path, device="cuda"):
    was_training = model.training
    model.to(device)
    model.eval()

    img = Image.open(img_path)
    img = data_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        print(f"Output shape: {outputs.shape}")
        print(f"Predictions shape: {preds.shape} ")

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)