"""
---
title: Train a ResNet on CIFAR 10
summary: >
  Train a ResNet on CIFAR 10
---

# Train a [ResNet](index.html) on CIFAR 10
"""
from typing import List, Optional

from torch import nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.resnet import ResNetBase
from torchvision import models


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # Number fo blocks for each feature map size
    n_blocks: List[int] = [3, 3, 3]
    # Number of channels for each feature map size
    n_channels: List[int] = [16, 32, 64]
    # Bottleneck sizes
    bottlenecks: Optional[List[int]] = None
    # Kernel size of the initial convolution layer
    first_kernel_size: int = 3


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


def main():
    # Create experiment
    experiment.create(name='resnet', comment='cifar10')
    # Create configurations
    conf = Configs()

    # Load configurations
    experiment.configs(conf, {
        'n_blocks': [3, 4, 23, 3],
        'n_channels': [64, 128, 256, 512],
        'bottlenecks': [64, 128, 256, 512],
        'first_kernel_size': 7,

        'optimizer.optimizer': 'SGD',
        'optimizer.learning_rate': 0.001,
        'optimizer.weight_decay': 0.0001,
        'optimizer.momentum': 0.9,

        'epochs': 10,
        'train_batch_size': 16,

        'train_dataset': 'cifar10_train_augmented',
        'valid_dataset': 'cifar10_valid_no_augment',
    })

    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})

    model = models.resnet101(pretrained=False)

    #experiment.add_pytorch_models({'model': model})

    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
