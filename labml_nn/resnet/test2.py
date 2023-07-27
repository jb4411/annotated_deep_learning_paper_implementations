import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Load the pretrained ResNet-101 model from torchvision
    model = models.resnet101(pretrained=False)

    # Specify the number of classes for your specific classification problem
    num_classes = 10  # replace with your number of classes

    # Modify the last layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    val_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    train_data = datasets.STL10('./data', split="train", download=True, transform=transforms.ToTensor())
    val_data = datasets.STL10('./data', split="test", download=True, transform=transforms.ToTensor())



    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    model = model.to(device)  # send model to device (CPU or GPU)

    # Number of epochs
    epochs = 2


    # Add two empty lists before the start of training
    train_losses = []
    val_errors = []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_losses.append(epoch_train_loss / len(train_loader))

        # Validate
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_error = 1 - (correct / total)
        val_errors.append(val_error)

        print(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]}, Validation Error: {val_errors[-1]}')

    # Plot the training loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the validation error
    plt.subplot(1, 2, 2)
    plt.plot(val_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
