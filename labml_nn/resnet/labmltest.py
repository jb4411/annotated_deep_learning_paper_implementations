from enum import Enum

import torch
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
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


def setup_dataset(dataset, batch_size=32):
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

def UNUSED_get_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)
    return model


def train_model(model, criterion, optimizer, num_epochs=10):
    writer = SummaryWriter()
    steps = 0
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


                # write to tensorboard
                if phase == 'train':
                    tracker.add('loss.train', running_loss)
                    tracker.add('accuracy.train', running_corrects/running_seen)
                else:
                    tracker.add('loss.valid', running_loss)
                    tracker.add('accuracy.valid', running_corrects / running_seen)
                steps += 1

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)


            tracker.save(epoch, {'loss.train': epoch_loss, 'accuracy.train': epoch_acc,
                                                      'loss.valid': valid_loss, 'accuracy.valid': (valid_correct / valid_total)})


            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            """# write to tensorboard
            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('training accuracy', epoch_acc, epoch)
            else:
                writer.add_scalar('validation loss', epoch_loss, epoch)
                writer.add_scalar('validation accuracy', epoch_acc, epoch)"""

        tracker.new_line()

    print('Training complete')


def main():
    # Batch size
    batch_size = 32
    # Number of epochs
    num_epochs = 3
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
    setup_dataset(DataSet.CIFAR10, batch_size)

    train_model(model, criterion, optimizer, num_epochs=num_epochs)


if __name__ == '__main__':
    main()
