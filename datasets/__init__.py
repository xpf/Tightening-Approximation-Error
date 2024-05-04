from os.path import join
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100

DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}


def build_data(data_name, data_path, train):
    data = DATASETS[data_name](root=join(data_path, data_name), train=train, transform=None)
    return data
