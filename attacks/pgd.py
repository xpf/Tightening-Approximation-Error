import torch, math
import torch.nn.functional as F


class PGD(object):
    def __init__(self, model, loader, n_iter=100, norm='linf', n_restarts=1, eps=None, n_class=10, target=-1, device='cuda:0'):
        super().__init__()
        self.model = model
        self.loader = loader
        self.n_iter = n_iter
        self.norm = norm
        self.eps = eps
        self.alpha_max = eps
        self.alpha_min = 0
        self.n_restarts = n_restarts
        self.n_class = n_class
        self.target = target
        self.device = device
        self.global_mask = torch.ones(size=(len(self.loader.dataset),), dtype=torch.bool).to(self.device)

    def update_loader(self, loader):
        self.loader = loader
        self.global_mask = torch.ones(size=(len(self.loader.dataset),), dtype=torch.bool).to(self.device)

    def update_n_iter(self, n_iter):
        self.n_iter = n_iter

    def update_n_restarts(self, n_restarts):
        self.n_restarts = n_restarts

    def global_adv_acc(self):
        adv_acc = torch.sum(self.global_mask).item() / self.global_mask.shape[0]
        return adv_acc

    def evaluate(self, loss):
        correct, total, invalid, acc = 0, 0, False, 2.0
        hard_mask = torch.ones(size=(len(self.loader.dataset),), dtype=torch.bool).to(self.device)
        for x, y, s in self.loader:
            x, y, s = x.to(self.device), y.to(self.device), s.to(self.device)
            q = F.one_hot(y, num_classes=self.n_class)
            if self.target != -1:
                t = F.one_hot(torch.zeros_like(y) + self.target, num_classes=self.n_class)
            ext_masks = torch.ones((x.shape[0],), dtype=torch.bool).to(self.device)
            ext_indxs = torch.arange(0, x.shape[0]).to(self.device).detach()
            if ext_indxs.shape[0] == 0: break
            for n_restart in range(self.n_restarts):
                x_clean = x[ext_masks, :, :, :].clone().detach()
                y_clean = y[ext_masks].clone().detach()
                q_clean = q[ext_masks, :].clone().detach()
                if self.target != -1:
                    t_clean = t[ext_masks, :].clone().detach()
                x_adv = x_clean.clone().detach()
                int_masks = torch.ones((x_adv.shape[0],), dtype=torch.bool).to(self.device).detach()
                int_indxs = torch.arange(0, x_adv.shape[0]).to(self.device).detach()
                if self.norm == 'linf':
                    n_adv = (torch.rand_like(x_adv) * 2 - 1).detach()
                    n_adv_norm = (n_adv.reshape([n_adv.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1])) + 1e-12
                    x_adv = (x_adv + self.eps * n_adv / n_adv_norm).clamp(0, 1).detach()
                elif self.norm == 'l2':
                    n_adv = torch.rand_like(x_adv).detach()
                    n_adv_norm = ((n_adv ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv = (x_adv + self.eps * n_adv / n_adv_norm).clamp(0, 1).detach()
                for i in range(self.n_iter):
                    x_adv.requires_grad = True
                    p = self.model(x_adv)
                    try:
                        if self.target == -1:
                            cost = torch.mean(torch.sum(loss(p, q_clean), dim=-1), dim=0)
                        else:
                            cost = -torch.mean(torch.sum(loss(p, t_clean), dim=-1), dim=0)
                        grad = torch.autograd.grad(cost, x_adv, retain_graph=False, create_graph=False)[0]
                    except:
                        invalid = True
                        break
                    if torch.sum(torch.isinf(grad.data)):
                        invalid = True
                        break
                    with torch.no_grad():
                        suc = torch.max(p, dim=1)[1] != y_clean
                        not_suc = torch.logical_not(suc)
                        int_masks[int_indxs[suc]] = False
                        int_indxs = int_indxs[not_suc]
                        if int_indxs.shape[0] == 0: break
                        x_adv, x_clean, y_clean, q_clean = x_adv[not_suc, :, :, :], x_clean[not_suc, :, :, :], y_clean[not_suc], q_clean[not_suc, :]
                        if self.target != -1:
                            t_clean = t_clean[not_suc, :]
                        grad = grad[not_suc, :, :, :]
                        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 + math.cos(i / self.n_iter * math.pi))
                        if self.norm == 'linf':
                            x_adv = x_adv + alpha * grad.sign()
                            delta = torch.clamp(x_adv - x_clean, min=-self.eps, max=self.eps)
                            x_adv = torch.clamp(x_clean + delta, min=0, max=1).detach()
                        elif self.norm == 'l2':
                            x_adv = x_adv + alpha * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                            delta = (x_adv - x_clean) / (((x_adv - x_clean) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                                self.eps * torch.ones_like(x_adv).detach(),
                                (((x_adv - x_clean) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                            )
                            x_adv = torch.clamp(x_clean + delta, min=0, max=1).detach()
                        if torch.sum(torch.isnan(x_adv.data)) > 0:
                            invalid = True
                            break
                if invalid: break
                suc = int_masks == False
                not_suc = torch.logical_not(suc)
                ext_masks[ext_indxs[suc]] = False
                ext_indxs = ext_indxs[not_suc]
                if ext_indxs.shape[0] == 0: break
            if invalid:
                acc = 2.0
            else:
                correct += torch.sum(ext_masks).item()
                total += ext_masks.shape[0]
                acc = correct / (total + 1e-12)
                hard_mask[s] = ext_masks
        self.global_mask = torch.logical_and(self.global_mask, hard_mask)
        hard_mask = hard_mask.cpu().detach().numpy()
        return acc, hard_mask


class OPGD(object):
    def __init__(self, model, loader, n_iter=100, alpha=4, norm='linf', n_restarts=1, eps=None, n_class=10, target=-1, device='cuda:0'):
        super().__init__()
        self.model = model
        self.loader = loader
        self.n_iter = n_iter
        self.norm = norm
        self.eps = eps
        self.alpha_max = eps
        self.alpha_min = 0
        self.n_restarts = n_restarts
        self.n_class = n_class
        self.target = target
        self.device = device
        self.alpha = (self.alpha_max - self.alpha_min) / alpha
        self.global_mask = torch.ones(size=(len(self.loader.dataset),), dtype=torch.bool).to(self.device)

    def update_loader(self, loader):
        self.loader = loader
        self.global_mask = torch.ones(size=(len(self.loader.dataset),), dtype=torch.bool).to(self.device)

    def update_n_iter(self, n_iter):
        self.n_iter = n_iter

    def update_n_restarts(self, n_restarts):
        self.n_restarts = n_restarts

    def global_adv_acc(self):
        adv_acc = torch.sum(self.global_mask).item() / self.global_mask.shape[0]
        return adv_acc

    def evaluate(self, loss):
        correct, total, invalid, acc = 0, 0, False, 2.0
        hard_mask = torch.ones(size=(len(self.loader.dataset),), dtype=torch.bool).to(self.device)
        for x, y, s in self.loader:
            x, y, s = x.to(self.device), y.to(self.device), s.to(self.device)
            q = F.one_hot(y, num_classes=self.n_class)
            if self.target != -1:
                t = F.one_hot(torch.zeros_like(y) + self.target, num_classes=self.n_class)
            ext_masks = torch.ones((x.shape[0],), dtype=torch.bool).to(self.device)
            ext_indxs = torch.arange(0, x.shape[0]).to(self.device).detach()
            if ext_indxs.shape[0] == 0: break
            for n_restart in range(self.n_restarts):
                x_clean = x[ext_masks, :, :, :].clone().detach()
                y_clean = y[ext_masks].clone().detach()
                q_clean = q[ext_masks, :].clone().detach()
                if self.target != -1:
                    t_clean = t[ext_masks, :].clone().detach()
                x_adv = x_clean.clone().detach()
                int_masks = torch.ones((x_adv.shape[0],), dtype=torch.bool).to(self.device).detach()
                int_indxs = torch.arange(0, x_adv.shape[0]).to(self.device).detach()
                if self.norm == 'linf':
                    n_adv = (torch.rand_like(x_adv) * 2 - 1).detach()
                    n_adv_norm = (n_adv.reshape([n_adv.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1])) + 1e-12
                    x_adv = (x_adv + self.eps * n_adv / n_adv_norm).clamp(0, 1).detach()
                elif self.norm == 'l2':
                    n_adv = torch.rand_like(x_adv).detach()
                    n_adv_norm = ((n_adv ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv = (x_adv + self.eps * n_adv / n_adv_norm).clamp(0, 1).detach()
                for i in range(self.n_iter):
                    x_adv.requires_grad = True
                    p = self.model(x_adv)
                    try:
                        if self.target == -1:
                            cost = torch.mean(torch.sum(loss(p, q_clean), dim=-1), dim=0)
                        else:
                            cost = -torch.mean(torch.sum(loss(p, t_clean), dim=-1), dim=0)
                        grad = torch.autograd.grad(cost, x_adv, retain_graph=False, create_graph=False)[0]
                    except:
                        invalid = True
                        break
                    if torch.sum(torch.isinf(grad.data)):
                        invalid = True
                        break
                    with torch.no_grad():
                        suc = torch.max(p, dim=1)[1] != y_clean
                        not_suc = torch.logical_not(suc)
                        int_masks[int_indxs[suc]] = False
                        int_indxs = int_indxs[not_suc]
                        if int_indxs.shape[0] == 0: break
                        x_adv, x_clean, y_clean, q_clean = x_adv[not_suc, :, :, :], x_clean[not_suc, :, :, :], y_clean[not_suc], q_clean[not_suc, :]
                        if self.target != -1:
                            t_clean = t_clean[not_suc, :]
                        grad = grad[not_suc, :, :, :]
                        # alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 + math.cos(i / self.n_iter * math.pi))
                        alpha = self.alpha
                        if self.norm == 'linf':
                            x_adv = x_adv + alpha * grad.sign()
                            delta = torch.clamp(x_adv - x_clean, min=-self.eps, max=self.eps)
                            x_adv = torch.clamp(x_clean + delta, min=0, max=1).detach()
                        elif self.norm == 'l2':
                            x_adv = x_adv + alpha * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                            delta = (x_adv - x_clean) / (((x_adv - x_clean) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                                self.eps * torch.ones_like(x_adv).detach(),
                                (((x_adv - x_clean) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                            )
                            x_adv = torch.clamp(x_clean + delta, min=0, max=1).detach()
                        if torch.sum(torch.isnan(x_adv.data)) > 0:
                            invalid = True
                            break
                if invalid: break
                suc = int_masks == False
                not_suc = torch.logical_not(suc)
                ext_masks[ext_indxs[suc]] = False
                ext_indxs = ext_indxs[not_suc]
                if ext_indxs.shape[0] == 0: break
            if invalid:
                acc = 2.0
            else:
                correct += torch.sum(ext_masks).item()
                total += ext_masks.shape[0]
                acc = correct / (total + 1e-12)
                hard_mask[s] = ext_masks
        self.global_mask = torch.logical_and(self.global_mask, hard_mask)
        hard_mask = hard_mask.cpu().detach().numpy()
        return acc, hard_mask
