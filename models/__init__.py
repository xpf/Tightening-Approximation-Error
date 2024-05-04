import torch, os
from collections import OrderedDict
from .resnet import ResNet18, PreActResNet, PreActBlockV2
from .wide_resnet import WideResNet

MODELS = {
    'cifar10': {
        'linf': {
            'Addepalli2021Towards_RN18': lambda: ResNet18(),
            'Cui2020Learnable_34_10': lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'Huang2020Self': lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
            'Sehwag2021Proxy_R18': lambda: ResNet18(),
            'Wu2020Adversarial': lambda: WideResNet(depth=34, widen_factor=10),
            'Zhang2019Theoretically': lambda: WideResNet(depth=34, widen_factor=10, sub_block1=True),
        },
        'l2': {
            'Sehwag2021Proxy_R18': lambda: ResNet18(),
            'Wu2020Adversarial': lambda: WideResNet(depth=34, widen_factor=10),
        }
    },
    'cifar100': {
        'linf': {
            'Addepalli2021Towards_PARN18': lambda: PreActResNet(PreActBlockV2, [2, 2, 2, 2], num_classes=100, bn_before_fc=True),
            'Cui2020Learnable_34_10_LBGAT0': lambda: WideResNet(depth=34, widen_factor=10, num_classes=100, sub_block1=True),
        }
    }
}


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def build_pretrained_model(data_name, norm, model_name, weight_path):
    model = MODELS[data_name][norm][model_name]()
    weight_filename = os.path.join(weight_path, data_name, norm, '{}.pt'.format(model_name))
    state_dict = torch.load(weight_filename, map_location=torch.device('cpu'))
    state_dict = rm_substr_from_state_dict(state_dict, 'module.')
    state_dict = rm_substr_from_state_dict(state_dict, 'model.')
    model.load_state_dict(state_dict, strict=True)
    return model
