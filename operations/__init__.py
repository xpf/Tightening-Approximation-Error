import torch


def add(x, y):
    return torch.add(x, y)


def mul(x, y):
    return torch.mul(x, y)


def exp(x):
    return torch.exp(x)


def sqrt(x):
    return torch.mul(torch.sign(x), torch.sqrt(torch.abs(x)))


def square(x):
    return torch.square(x)


def abs(x):
    return torch.abs(x)


def neg(x):
    return torch.neg(x)


def inv(x):
    return torch.div(torch.sign(x), (torch.abs(x) + 1e-12))


def log(x):
    return torch.mul(torch.sign(x), torch.log(torch.abs(x) + 1e-12))


def max(x):
    return torch.max(x, dim=-1, keepdim=True)[0]


def sum(x):
    return torch.sum(x, dim=-1, keepdim=True)


def softmax(x):
    return torch.softmax(x, dim=-1)


def sort(x):
    return torch.sort(x, dim=-1, descending=True)[0]


def ce_loss(x, y):
    return neg(mul(log(softmax(x)), y))


def cw_loss(x, y):
    return add(max(mul(x, add(1, neg(y)))), neg(sum(mul(x, y))))
