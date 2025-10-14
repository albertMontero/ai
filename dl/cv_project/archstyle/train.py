import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tempfile import TemporaryDirectory

from PIL import Image
from torch.optim import lr_scheduler

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def train_model(model, dataloaders, criterion=None, optimizer=None, scheduler=None, num_epochs=5):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if scheduler is None:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.to(device)

    since = time.time()
    dataset_sizes = {'train': len(dataloaders["train"].dataset), 'val': len(dataloaders["val"].dataset)}
    history = {
        'train_loss': [0] * num_epochs,
        'train_accuracy': [0] * num_epochs,
        'val_loss': [0] * num_epochs,
        'val_accuracy': [0] * num_epochs
    }

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                history[f"{phase}_loss"][epoch] = epoch_loss
                history[f'{phase}_accuracy'][epoch] = epoch_acc.cpu().numpy()

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model, history


def evaluate(model, dataloader):

    was_training = model.training
    model.to(device)
    model.eval()
    
    # dataloader = DataLoader(ds, batch_size=32, shuffle=False)
    
    running_corrects = 0
    total = 0
    
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels).float()
            total += labels.size(0)
    
    accuracy = running_corrects.double() / total
    print(f'Accuracy: {accuracy:.4f}')
    print("Total: ", total)
    print("Correct: ", running_corrects.cpu().numpy())

    model.train(mode=was_training)
