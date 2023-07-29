import math
import time
from enum import Enum

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from labml import tracker, experiment, monit
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet
from typing import List, Any, Type, Union

tracker.set_scalar("loss.*", True)
tracker.set_scalar("accuracy.*", True)


class DataSet(Enum):
    CIFAR10 = 1
    STL10 = 2


class StepType(Enum):
    PERF_STEP = 1
    APPROX_STEP = 2
    APPROX_DATA_STEP = 3
    PERF_DATA_STEP = 4


class Phase(Enum):
    TRAIN = 1
    VALID = 2


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global data_loaders
global train_data
global val_data
global batch_len
step_type: StepType = StepType.PERF_DATA_STEP
global train_steps
global valid_steps
global t_step
global v_step
global t_count
global v_count


def setup_dataset(dataset: DataSet, batch_size=32):
    global batch_len
    batch_len = batch_size
    global data_loaders
    global train_data
    global val_data
    if dataset == DataSet.CIFAR10:
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        val_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == DataSet.STL10:
        train_data = datasets.STL10('./data', split="train", download=True, transform=transforms.ToTensor())
        val_data = datasets.STL10('./data', split="test", download=True, transform=transforms.ToTensor())

    # Training and validation data loaders
    data_loaders = {
        Phase.TRAIN: torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
        Phase.VALID: torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    }

    return data_loaders


def setup_steps():
    global train_steps
    global valid_steps
    global t_step
    global v_step

    train_steps = 0
    valid_steps = 0
    if step_type == StepType.PERF_STEP:
        t_len = len(data_loaders[Phase.TRAIN])
        v_len = len(data_loaders[Phase.VALID])
        lcm = math.lcm(t_len, v_len)
        t_step = int(lcm / t_len)
        v_step = int(lcm / v_len)

        def inc(phase: Phase, epoch):
            global train_steps
            global valid_steps
            if phase == Phase.TRAIN:
                train_steps += t_step
                return train_steps
            else:
                valid_steps += v_step
                return valid_steps

        return inc
    elif step_type == StepType.APPROX_STEP:
        t_len = len(data_loaders[Phase.TRAIN])
        v_len = len(data_loaders[Phase.VALID])
        if t_len > v_len:
            t_step = 1
            v_step = t_len / v_len
        else:
            t_step = v_len / t_len
            v_step = 1

        def inc(phase: Phase, epoch):
            global train_steps
            global valid_steps
            if phase == Phase.TRAIN:
                train_steps += t_step
                return int(train_steps)
            else:
                valid_steps += v_step
                return int(valid_steps)

        return inc
    elif step_type == StepType.APPROX_DATA_STEP:
        t_len = len(train_data)
        v_len = len(val_data)
        if t_len > v_len:
            t_step = batch_len
            v_step = batch_len * (t_len / v_len)
        else:
            t_step = batch_len * (v_len / t_len)
            v_step = batch_len

        def inc(phase: Phase, epoch):
            global train_steps
            global valid_steps
            if phase == Phase.TRAIN:
                train_steps += t_step
                return int(train_steps)
            else:
                valid_steps += v_step
                return int(valid_steps)

        return inc
    elif step_type == StepType.PERF_DATA_STEP:
        t_len = len(train_data)
        v_len = len(val_data)
        target = max(t_len, v_len)
        if t_len > v_len:
            t_step = batch_len
            v_step = batch_len * (t_len / v_len)
        else:
            t_step = batch_len * (v_len / t_len)
            v_step = batch_len
        global t_count
        global v_count
        t_count = 0
        v_count = 0
        t_batch = len(data_loaders[Phase.TRAIN])
        v_batch = len(data_loaders[Phase.VALID])

        def inc(phase: Phase, epoch):
            global train_steps
            global valid_steps
            global t_count
            global v_count

            if phase == Phase.TRAIN:
                t_count += 1
                train_steps += t_step
                if t_count >= t_batch:
                    t_count = 0
                    train_steps = target * (epoch + 1)
                return int(train_steps)
            else:
                v_count += 1
                valid_steps += v_step
                if v_count >= v_batch:
                    v_count = 0
                    valid_steps = target * (epoch + 1)
                return int(valid_steps)

        return inc


def train_model(model, criterion, optimizer, num_epochs=10, save_per_epoch=False, name="sample"):
    tracked_data = {
        'loss.train': [],
        'accuracy.train': [],
        'loss.valid': [],
        'accuracy.valid': []
    }

    inc = setup_steps()

    with experiment.record(name=name, token='http://localhost:5005/api/v1/track?'):
        for epoch in monit.loop(range(num_epochs)):
            # print(f"Epoch {epoch}/{num_epochs}")
            # print('-' * 10)

            for phase in [Phase.TRAIN, Phase.VALID]:
                if phase == Phase.TRAIN:
                    text = "Train"
                    model.train()
                else:
                    text = "Valid"
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_seen = 0

                for inputs, labels in monit.iterate(text, data_loaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == Phase.TRAIN):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == Phase.TRAIN:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_seen += len(labels.data)
                    if save_per_epoch:
                        if phase == Phase.TRAIN:
                            train_loss = running_loss / running_seen
                            train_acc = running_corrects / running_seen
                            tracked_data['loss.train'].append(train_loss)
                            tracked_data['accuracy.train'].append(train_acc)

                        else:
                            valid_loss = running_loss / running_seen
                            valid_acc = running_corrects / running_seen
                            tracked_data['loss.valid'].append(valid_loss)
                            tracked_data['accuracy.valid'].append(valid_acc)
                    else:
                        if phase == Phase.TRAIN:
                            train_loss = running_loss / running_seen
                            train_acc = running_corrects / running_seen
                            tracker.save(inc(phase, epoch), {'loss.train': train_loss, 'accuracy.train': train_acc})
                        else:
                            valid_loss = running_loss / running_seen
                            valid_acc = running_corrects / running_seen
                            tracker.save(inc(phase, epoch), {'loss.valid': valid_loss, 'accuracy.valid': valid_acc})

                if save_per_epoch:
                    if phase == Phase.TRAIN:
                        for i in range(len(tracked_data['loss.train'])):
                            tracker.save(inc(phase, epoch), {'loss.train': tracked_data['loss.train'][i],
                                                             'accuracy.train': tracked_data['accuracy.train'][i]})
                        tracked_data['loss.train'] = []
                        tracked_data['accuracy.train'] = []
                    else:
                        for i in range(len(tracked_data['loss.valid'])):
                            tracker.save(inc(phase, epoch), {'loss.valid': tracked_data['loss.valid'][i],
                                                             'accuracy.valid': tracked_data['accuracy.valid'][i]})
                        tracked_data['loss.valid'] = []
                        tracked_data['accuracy.valid'] = []

            tracker.new_line()

    print('Training complete')


def generate_resnet(layers: List[int], num_classes: int = 10, block_type: Type[Union[BasicBlock, Bottleneck]] = None,
                    **kwargs: Any) -> ResNet:
    """Generate a ResNet model with an arbitrary number of layers.

    Args:
        layers (List[int]): A list that contains the number of blocks for each layer.
        num_classes (int, optional): The number of classes for the classification task. Defaults to 1000.
        block_type (Type[Union[BasicBlock, Bottleneck]]): The type of block to use
    Returns:
        ResNet: The ResNet model.
    """
    if block_type is None:
        # If the number of blocks is less than or equal to 2, use BasicBlock, else use Bottleneck
        block = BasicBlock if max(layers) <= 2 else Bottleneck
    else:
        block = block_type

    # Create the ResNet model
    model = _resnet(block, layers, num_classes=num_classes, **kwargs)

    return model


def calculate_total_layers(layers: List[int], block: Type[Union[BasicBlock, Bottleneck]]) -> int:
    """Calculate the total number of layers in a ResNet model.

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): The block type, either BasicBlock or Bottleneck.
        layers (List[int]): A list that contains the number of blocks for each layer.

    Returns:
        int: The total number of layers in the ResNet model.
    """
    # Count of layers in a block
    if block == BasicBlock:
        block_layers = 2
    else:
        block_layers = 3

    # Count the total number of layers in blocks
    total_layers = sum(block_layers * x for x in layers)

    # Add 1 for the initial convolutional layer, 1 for the max pooling layer, and 1 for the final fully connected layer
    total_layers += 3

    return total_layers


def get_model(num_layers, num_classes=10, block_type: Type[Union[BasicBlock, Bottleneck]] = None,
              **kwargs: Any) -> ResNet:
    if num_layers in [18, 34, 50, 101, 152]:
        pre_defined = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }
        model = pre_defined[num_layers](num_classes=num_classes, **kwargs)
    else:
        if block_type is None:
            if num_layers < 50:
                block_type = BasicBlock
                block_layers = 2
            else:
                block_type = Bottleneck
                block_layers = 3
        else:
            if block_type == BasicBlock:
                block_layers = 2
            else:
                block_layers = 3

        target = num_layers - 3
        offset = target % block_layers
        target += block_layers - offset
        target -= block_layers * 2
        """
         50 -> 4, 6  ---  29 -> 12, 18
        101 -> 4, 23 ---  80 -> 12, 69
        152 -> 8, 36 --- 131 -> 24, 108
        """
        mid = target // 2
        layers = [block_layers, mid, target - mid, block_layers]
        actual = calculate_total_layers(layers, block_type)
        if num_layers != actual:
            print(f"Target = {num_layers} layers, actual = {actual} layers.")
        model = generate_resnet(layers, num_classes=num_classes, block_type=block_type, **kwargs)

    model = model.to(device)
    return model


def main():
    # Number of layers for the resnet model
    num_layers = 50
    # Dataset
    dataset: DataSet = DataSet.CIFAR10
    # Batch size
    batch_size = 32
    # Number of epochs
    num_epochs = 5
    # Learning rate
    lr = 0.01
    # Optimizer momentum
    momentum = 0.9

    # metric save method
    save_per_epoch = False
    # Metric step type
    global step_type
    step_type = StepType.PERF_DATA_STEP

    # model
    model = get_model(num_layers)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Load dataset
    setup_dataset(dataset, batch_size)

    start = time.perf_counter()
    train_model(model, criterion, optimizer, num_epochs=num_epochs, save_per_epoch=save_per_epoch,
                name=f"ResNet - {dataset}")
    end = time.perf_counter()
    print(f"Training took {end - start}")


if __name__ == '__main__':
    main()
