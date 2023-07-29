import math
from enum import Enum

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from labml import tracker, experiment

tracker.set_scalar("loss.*", True)
tracker.set_scalar("accuracy.*", True)


class DataSet(Enum):
    CIFAR10 = 1
    STL10 = 2


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global data_loaders


def setup_dataset(dataset: DataSet, batch_size=32):
    if dataset == DataSet.CIFAR10:
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        val_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == DataSet.STL10:
        train_data = datasets.STL10('./data', split="train", download=True, transform=transforms.ToTensor())
        val_data = datasets.STL10('./data', split="test", download=True, transform=transforms.ToTensor())

    global data_loaders
    # Training and validation data loaders
    data_loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    }

    return data_loaders


def train_model(model, criterion, optimizer, num_epochs=10):
    save_per_epoch = False

    train_steps = 0
    valid_steps = 0

    t_len = len(data_loaders['train'])
    v_len = len(data_loaders['val'])
    lcm = math.lcm(t_len, v_len)
    t_step = int(lcm / t_len)
    v_step = int(lcm / v_len)


    with experiment.record(name='sample', token='http://localhost:5005/api/v1/track?'):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs}")
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_seen = 0

                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_seen += len(labels.data)

                    if save_per_epoch:
                        if phase == 'train':
                            tracker.add('loss.train', running_loss / running_seen)
                            tracker.add('accuracy.train', running_corrects / running_seen)
                        else:
                            tracker.add('loss.valid', running_loss / running_seen)
                            tracker.add('accuracy.valid', running_corrects / running_seen)
                    else:
                        if phase == 'train':
                            train_loss = running_loss / running_seen
                            train_acc = running_corrects / running_seen
                            tracker.save(train_steps, {'loss.train': train_loss, 'accuracy.train': train_acc})
                            train_steps += t_step
                        else:
                            valid_loss = running_loss / running_seen
                            valid_acc = running_corrects / running_seen
                            tracker.save(valid_steps, {'loss.valid': valid_loss, 'accuracy.valid': valid_acc})
                            valid_steps += v_step

                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                epoch_acc = running_corrects / len(data_loaders[phase].dataset)
                if save_per_epoch:
                    if phase == 'train':
                        train_loss = epoch_loss
                        train_acc = epoch_acc
                        tracker.save(epoch, {'loss.train': train_loss, 'accuracy.train': train_acc})
                    else:
                        valid_loss = epoch_loss
                        valid_acc = epoch_acc
                        tracker.save(epoch, {'loss.train': train_loss, 'accuracy.train': train_acc,
                                             'loss.valid': valid_loss, 'accuracy.valid': valid_acc})

            tracker.new_line()

    print('Training complete')


def main():
    # Batch size
    batch_size = 32
    # Number of epochs
    num_epochs = 4
    # Learning rate
    lr = 0.001
    # Optimizer momentum
    momentum = 0.9

    # model
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Load dataset
    setup_dataset(DataSet.STL10, batch_size)

    train_model(model, criterion, optimizer, num_epochs=num_epochs)


if __name__ == '__main__':
    main()
