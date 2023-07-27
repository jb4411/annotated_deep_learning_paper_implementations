import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from labml import tracker, experiment


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tracker.set_scalar("loss.*", True)
tracker.set_scalar("accuracy.*", True)

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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    model = model.to(device)  # send model to device (CPU or GPU)

    # Number of epochs
    epochs = 2

    with experiment.record(name='sample', token='http://localhost:5005/api/v1/track?'):
        for epoch in range(epochs):
            # Train
            train_total = 0
            train_correct = 0
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # send data to device
                optimizer.zero_grad()
                outputs = model(inputs)
                train_loss = criterion(outputs, targets)
                train_loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                #tracker.save(epoch, {'loss.train': train_loss, 'accuracy.train': (train_correct / train_total)})
                tracker.add("loss.", criterion(outputs, targets))
                tracker.save()

            # Validate
            model.eval()
            valid_total = 0
            valid_correct = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)  # send data to device
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    valid_total += targets.size(0)
                    valid_correct += (predicted == targets).sum().item()

                    valid_loss = criterion(outputs, targets)

                    #tracker.save()
                    #tracker.save(epoch, {'loss.train': train_loss, 'accuracy.train': (train_correct / train_total),
                    #                     'loss.valid': valid_loss, 'accuracy.valid': (valid_correct / valid_total)})
            #print(f'Epoch: {epoch + 1}/{epochs}, Accuracy: {(correct / total) * 100}%')
            tracker.new_line()

if __name__ == '__main__':
    main()
