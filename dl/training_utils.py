import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryDirectory

from torch.optim import lr_scheduler

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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

def evaluate(model, ds):
    model.eval()
    with torch.inference_mode():
        model.to("cpu")
        pred = model(ds.data.float())
        accuracy = (torch.argmax(pred, dim=1) == ds.targets).float()
        print(f'Test accuracy: {accuracy.float().mean():.4f}')
    model.train()


# simple training loop
# TODO: add early stopping, save model, etc.

def train(model, num_epochs, train_dl, valid_dl, loss_fn=None, optimizer=None):
    history = {
        'train_loss': [0] * num_epochs,
        'train_accuracy': [0] * num_epochs,
        'val_loss': [0] * num_epochs,
        'val_accuracy': [0] * num_epochs
    }

    model.to(device)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)

            # Handle spatial outputs from CNNs (apply global pooling if needed)
            if len(pred.shape) == 4:  # [batch, classes, height, width]
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, 1)  # [batch, classes, 1, 1]
                pred = pred.view(pred.size(0), -1)  # [batch, classes]
            
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            history['train_loss'][epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            history['train_accuracy'][epoch] += is_correct.sum().cpu()

        history['train_loss'][epoch] /= len(train_dl.dataset)
        history['train_accuracy'][epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                
                # Handle spatial outputs from CNNs
                if len(pred.shape) == 4:
                    pred = torch.nn.functional.adaptive_avg_pool2d(pred, 1)
                    pred = pred.view(pred.size(0), -1)
                
                loss = loss_fn(pred, y_batch)
                history['val_loss'][epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                history['val_accuracy'][epoch] += is_correct.sum().cpu()

        history['val_loss'][epoch] /= len(valid_dl.dataset)
        history['val_accuracy'][epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1}: train_acc: {history["train_accuracy"][epoch]:.4f} val_acc: {history["val_accuracy"][epoch]:.4f} train_loss: {history["train_loss"][epoch]:.4f} val_loss: {history["val_loss"][epoch]:.4f}')

    return history
