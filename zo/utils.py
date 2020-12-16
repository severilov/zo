import torch

from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset


def get_data():
    """
    Make MNIST datasets for training and testing
    :return: train_data, valid_data, test_data
    """
    data = MNIST('mnist', download=True, train=True)
    train_data = TensorDataset(data.train_data.view(-1, 28 * 28).float() / 255)
    data = MNIST('mnist', download=True, train=False)
    test_data_raw = TensorDataset(data.test_data.view(-1, 28 * 28).float() / 255)

    train_data.tensors = (nn.AvgPool2d(2, 2)(train_data.tensors[0].view(-1, 28, 28)).data.view(-1, 196),)
    test_data_raw.tensors = (nn.AvgPool2d(2, 2)(test_data_raw.tensors[0].view(-1, 28, 28)).data.view(-1, 196),)

    valid_data = TensorDataset(test_data_raw.tensors[0][:5000])
    test_data = TensorDataset(test_data_raw.tensors[0][5000:])

    return train_data, valid_data, test_data


def generate_many_samples(model, num_samples, batch_size):
    """
    Generate samples for likelihood estimation
    :param model:
    :param num_samples:
    :param batch_size:
    :return: samples
    """
    size = 0
    res = []
    while size < num_samples:
        res.append(model.generate_samples(min(batch_size, num_samples - size)))
        size += batch_size
    return torch.cat(res, 0)


def timer(start, end):
    """
    Print execution time in convenient format
    :param start: start time, float
    :param end: end time, float
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}h:{:0>2}m:{:.0f}s".format(int(hours), int(minutes), seconds))
