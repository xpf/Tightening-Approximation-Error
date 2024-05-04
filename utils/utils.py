import torch, numpy as np


def data_filtering(model, data, loader, device):
    acc_masks = torch.zeros((data.data.shape[0],), dtype=torch.bool).to(device)
    for x, y, s in loader:
        with torch.no_grad():
            x, y, s = x.to(device), y.to(device), s.to(device)
            p = (torch.max(model(x), dim=1)[1] == y)
            acc_masks[s] = p
    acc_masks = acc_masks.cpu().detach().numpy()
    data.data = data.data[acc_masks, :, :, :]
    data.targets = list(np.array(data.targets)[acc_masks])
    correct, total = np.sum(acc_masks), acc_masks.shape[0]
    return data, correct / (total + 1e-12)


def get_name(opts):
    name = '{}_{}_{}_{}_{}_{}_{}'.format(
        opts.data_name, opts.model_name, opts.norm, opts.data_num, opts.n_iter, opts.target, opts.space
    ).lower()
    return name
