import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.optim as optim
from enum import Enum


class DataSet(Enum):
    CIFAR10 = 1
    STL10 = 2


def setup_dataset(dataset):
    if dataset == DataSet.CIFAR10:
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        val_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == DataSet.STL10:
        train_data = datasets.STL10('./data', split="train", download=True, transform=transforms.ToTensor())
        val_data = datasets.STL10('./data', split="test", download=True, transform=transforms.ToTensor())

    # Training and validation data loaders
    data_loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True),
        'val': torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
    }

    return data_loaders


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = models.resnet101(pretrained=False, num_classes=10)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # torch.optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    data_loaders = setup_dataset(DataSet.STL10)

    # Number of epochs
    num_epochs = 3

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
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

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


if __name__ == '__main__':
    main()
