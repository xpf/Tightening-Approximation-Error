import time, random, torch.nn as nn
import argparse, os, json, tqdm
from deap import gp, creator, base, tools
from attacks.apgd import APGDAttack
from robustbench.utils import load_model
from torch.utils.data import DataLoader
from datasets import build_data
from utils.utils import data_filtering
from attacks.pgd import PGD
from operations import *

pset = gp.PrimitiveSet('Search', 2)
pset.addPrimitive(add, 2), pset.addPrimitive(mul, 2), pset.addPrimitive(exp, 1)
pset.addPrimitive(sqrt, 1), pset.addPrimitive(square, 1), pset.addPrimitive(abs, 1), pset.addPrimitive(neg, 1), pset.addPrimitive(inv, 1)
pset.addPrimitive(log, 1), pset.addPrimitive(max, 1), pset.addPrimitive(sum, 1), pset.addPrimitive(softmax, 1)
pset.addEphemeralConstant('rand', lambda: random.randint(0, 1))
pset.renameArguments(ARG0='p', ARG1='q')
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=7)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)


def bs1(p, q):
    return exp(10 * softmax(p) / max(softmax(p)))


def bs2(p, q):
    return exp(- max(softmax(p + 2 * softmax(5 * p))))


def bs3(p, q):
    return softmax(-softmax(exp(p) * 2 * p)) * (softmax(2 * p) + 2 * q)


def bs4(p, q):
    return square(softmax(softmax(2 * p) - q + p) - q)


def bs5(p, q):
    return exp(- max(softmax(exp(softmax(exp(p) + p) + 1) + p) + 1))


def bs6(p, q):
    return inv(sum(softmax(sqrt(6 * p * p * p)) * (1 + q)))


def attack(opts):
    norm, n_iter, data_num, device = 'linf', 100, -1, 'cuda:0'
    name = '{}_{:.4f}'.format(opts.model_name, opts.eps)
    print('attack', name)

    model = load_model(model_name=opts.model_name, model_dir='./weights', norm='Linf').eval()
    # model = nn.DataParallel(model, device_ids=[0, 1, 2]).to(device)
    model = model.to(device)

    val_data = build_data('cifar10', '/data/xpf/datasets', False)
    val_loader = DataLoader(dataset=val_data, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    evaluations = {}
    print('step 0: calculate clean accuracy ...')
    start_time = time.time()
    val_data, clean_acc = data_filtering(model, val_data, val_loader, device)
    evaluations['clean_acc'] = clean_acc
    print('        clean accuracy: {:.4f}, time elapsed: {:.2f}.'.format(clean_acc, time.time() - start_time))
    if data_num != -1:
        evaluations['clean_acc'] = 1.0
        val_data.data = val_data.data[:data_num, :, :, :]
        val_data.targets = val_data.targets[:data_num]
    val_loader = DataLoader(dataset=val_data, batch_size=opts.batch_size, shuffle=False, num_workers=2)

    attacker = PGD(model, val_loader, n_iter, norm, 1, opts.eps, 10, -1, device)
    print('step 1: evaluate LPGD + CE & CW baselines ...')
    start_time = time.time()
    adv_acc, hard_mask = attacker.evaluate(ce_loss)
    evaluations['adv_acc_ce_lpgd'] = adv_acc
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    start_time = time.time()
    adv_acc, hard_mask = attacker.evaluate(cw_loss)
    evaluations['adv_acc_cw_lpgd'] = adv_acc
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    autoattacker = APGDAttack(model, n_iter=n_iter, norm='Linf' if norm == 'linf' else 'L2', n_restarts=1, eps=opts.eps, loss='ce', device=device, verbose=False)
    print('step 2: evaluate APGD + CE & DLR baselines ...')
    start_time = time.time()
    correct, totoal = 0, 0
    for x, y, _ in val_loader:
        x, y = x.to(device), y.to(device)
        acc, _ = autoattacker.perturb(x, y)
        correct += acc.sum().item()
        totoal += acc.shape[0]
    adv_acc = correct / totoal
    evaluations['adv_acc_ce_apgd'] = adv_acc
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    autoattacker = APGDAttack(model, n_iter=n_iter, norm='Linf' if norm == 'linf' else 'L2', n_restarts=1, eps=opts.eps, loss='dlr', device=device, verbose=False)
    start_time = time.time()
    correct, totoal = 0, 0
    for x, y, _ in val_loader:
        x, y = x.to(device), y.to(device)
        acc, _ = autoattacker.perturb(x, y)
        correct += acc.sum().item()
        totoal += acc.shape[0]
    adv_acc = correct / totoal
    evaluations['adv_acc_dlr_apgd'] = adv_acc
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    for i, bs in enumerate([bs1, bs2, bs3, bs4, bs5]):
        start_time = time.time()
        adv_acc, hard_mask = attacker.evaluate(bs)
        evaluations['adv_acc_bs{}_lpgd'.format(i)] = adv_acc
        print('        searched adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    with open(os.path.join(opts.loss_path, name), 'w') as fp:
        json.dump(evaluations, fp)

    autoattacker = APGDAttack(model, n_iter=n_iter, norm='Linf' if norm == 'linf' else 'L2', n_restarts=1, eps=opts.eps, loss='searched', device=device, verbose=False)
    start_time = time.time()
    correct, totoal = 0, 0
    run_tqdm = tqdm.tqdm(val_loader)
    for x, y, _ in run_tqdm:
        x, y = x.to(device), y.to(device)
        acc, _ = autoattacker.perturb(x, y, criterion=cw_loss)
        correct += acc.sum().item()
        totoal += acc.shape[0]
        run_tqdm.set_description('{:.4f}'.format(clean_acc * correct / totoal))
    adv_acc = correct / totoal
    evaluations['adv_acc_cw_apgd'] = adv_acc
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    autoattacker = APGDAttack(model, n_iter=n_iter, norm='Linf' if norm == 'linf' else 'L2', n_restarts=1, eps=opts.eps, loss='searched', device=device, verbose=False)
    for i, bs in enumerate([bs1, bs2, bs3, bs4, bs5]):
        start_time = time.time()
        correct, totoal = 0, 0
        run_tqdm = tqdm.tqdm(val_loader)
        for x, y, _ in run_tqdm:
            x, y = x.to(device), y.to(device)
            acc, _ = autoattacker.perturb(x, y, criterion=bs)
            correct += acc.sum().item()
            totoal += acc.shape[0]
            run_tqdm.set_description('{:.4f}'.format(clean_acc * correct / totoal))
        adv_acc = correct / totoal
        evaluations['adv_acc_bs{}_apgd'.format(i)] = adv_acc
        print('        searched adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc * clean_acc, time.time() - start_time))

    with open(os.path.join(opts.loss_path, name + '_apgd'), 'w') as fp:
        json.dump(evaluations, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_path', type=str, default='./losses')

    parser.add_argument('--model_name', type=str, default='Wang2020Improving')
    parser.add_argument('--eps', type=float, default=12 / 255)
    parser.add_argument('--batch_size', type=int, default=128)

    opts = parser.parse_args()

    for opts.eps in [14 / 255]:
        attack(opts)
