"""
---
title: Train a ResNet on CIFAR 10
summary: >
  Train a ResNet on CIFAR 10
---

# Train a [ResNet](index.html) on CIFAR 10
"""
from enum import Enum
from typing import List, Optional

from torch import nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.resnet import ResNetBase
from torchvision import models
from stl10 import STL10Configs
from labml_nn.experiments.mnist import MNISTConfigs
from torchvision import datasets

class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # Number of blocks for each feature map size
    n_blocks: List[int] = [3, 3, 3]
    # Number of channels for each feature map size
    n_channels: List[int] = [16, 32, 64]
    # Bottleneck sizes
    bottlenecks: Optional[List[int]] = None
    # Kernel size of the initial convolution layer
    first_kernel_size: int = 3


class STLConfigs(Configs):
    train_dataset: datasets.STL10
    valid_dataset: datasets.STL10


@option(Configs.model)
def _resnet(c: Configs):
    """
    ### Create model
    """
    # [ResNet](index.html)
    base = ResNetBase(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=3, first_kernel_size=c.first_kernel_size)
    # Linear layer for classification
    classification = nn.Linear(c.n_channels[-1], 10)

    # Stack them
    model = nn.Sequential(base, classification)
    # Move the model to the device
    return model.to(c.device)


class Dataset(Enum):
    CIFAR10 = 1
    STL10 = 2


def main():
    dataset: Dataset
    dataset = Dataset.STL10

    # Create experiment
    experiment.create(name='resnet', comment=f"{dataset}")

    # Create configurations
    if dataset == Dataset.CIFAR10:
        train_dataset = "cifar10_train_augmented"
        valid_dataset = "cifar10_valid_no_augment"
        conf = Configs()
    else:
        train_dataset = "stl10_train_dataset"
        valid_dataset = "stl10_valid_dataset"
        train_dataset = "cifar10_train_augmented"
        valid_dataset = "cifar10_valid_no_augment"
        conf = STLConfigs()

    # Load configurations
    experiment.configs(conf, {
        'n_blocks': [200, 200, 200],
        'first_kernel_size': 7,

        'optimizer.optimizer': 'SGD',
        'optimizer.learning_rate': 0.001,
        'optimizer.weight_decay': 0.0001,
        'optimizer.momentum': 0.9,

        'epochs': 3,
        'train_batch_size': 32,

        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset,
    })

    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})

    #model = models.resnet101(pretrained=False)
    #model = models.resnet18(pretrained=True)
    #experiment.add_pytorch_models({'model': model})

    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
