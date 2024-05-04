import time, random, os, pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from deap import gp, creator, base, tools
from functools import reduce
from autoattack import AutoAttack

from opts import get_opts
from datasets import build_data
from models import build_pretrained_model
from utils.utils import data_filtering, get_name
from attacks.pgd import PGD
from operations import *
from gp.algorithm import eaSimple

import warnings

warnings.filterwarnings("ignore")


def main(opts):
    name = get_name(opts)
    print('------- autoloss-ar: {}, suffix: {} -------'.format(name, opts.suffix))
    model = build_pretrained_model(opts.data_name, opts.norm, opts.model_name, opts.weight_path).eval()
    device_num = torch.cuda.device_count()
    model = model.to(opts.device) if device_num == 1 else nn.DataParallel(model, device_ids=list(range(device_num))).to(opts.device)
    val_data = build_data(opts.data_name, opts.data_path, False)
    batch_size = opts.batch_size * device_num
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    print('step 0: calculate clean accuracy ...')
    start_time = time.time()
    val_data, clean_acc = data_filtering(model, val_data, val_loader, opts.device)
    print('        clean accuracy: {:.4f}, time elapsed: {:.2f}.'.format(clean_acc, time.time() - start_time))

    if opts.target != -1:
        target_filter = np.array(val_data.targets) != opts.target
        val_data.data = val_data.data[target_filter, :, :, :]
        val_data.targets = list(np.array(val_data.targets)[target_filter])

    val_data.data = val_data.data[:opts.data_num, :, :, :]
    val_data.targets = val_data.targets[:opts.data_num]

    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    attacker = PGD(model, val_loader, opts.n_iter, opts.norm, opts.n_restarts, opts.eps, opts.n_class, opts.target,
                   opts.device)
    print('step 1: evaluate baseline ...')
    start_time = time.time()
    adv_acc, hard_mask = attacker.evaluate(ce_loss)
    attacker.update_loader(val_loader)
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc, time.time() - start_time))
    start_time = time.time()
    adv_acc, hard_mask = attacker.evaluate(cw_loss)
    attacker.update_loader(val_loader)
    print('        baseline adversarial accuracy: {:.4f}, time elapsed: {:.2f}.'.format(adv_acc, time.time() - start_time))

    print('step 2: create GP search ...')
    pset = gp.PrimitiveSet('Search', 2)
    pset.addPrimitive(add, 2), pset.addPrimitive(mul, 2), pset.addPrimitive(exp, 1)
    pset.addPrimitive(sqrt, 1), pset.addPrimitive(square, 1), pset.addPrimitive(abs, 1), pset.addPrimitive(neg, 1)
    pset.addPrimitive(inv, 1), pset.addPrimitive(log, 1)
    if opts.space == 0:
        pset.addPrimitive(softmax, 1), pset.addPrimitive(max, 1), pset.addPrimitive(sum, 1)
    pset.addEphemeralConstant('rand', lambda: random.randint(0, 1))
    pset.renameArguments(ARG0='p', ARG1='q')
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=7)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)

    def evalRobustness(individual):
        func = toolbox.compile(expr=individual)
        error, hard_mask = attacker.evaluate(func)
        return error, hard_mask

    toolbox.register('evaluate', evalRobustness)
    toolbox.register('select', tools.selLexicase)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate('mate', gp.staticLimit(key=len, max_value=25))
    toolbox.decorate('mutate', gp.staticLimit(key=len, max_value=25))
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register('min', np.min)

    print('step 3: run GP search ...')
    pop = toolbox.population(n=opts.n_pop)
    hof = tools.HallOfFame(1)
    pop, logbook, records = eaSimple(pop, toolbox, opts.cx_pb, opts.mut_pb, opts.n_gen, stats=mstats, halloffame=hof, verbose=True, attacker=attacker)

    print('step 4: nms for compression ...')
    records = reduce(lambda x, y: x + y, records)
    records = sorted(records, key=lambda x: x[0].fitness.values[0])
    keep_records, mask = [records[0]], records[0][1]
    del records[0]
    while True:
        if len(records) == 0: break
        diffs = []
        for i in range(len(records)):
            diffs.append(np.sum(mask != np.logical_and(mask, records[i][1])))
        diffs = np.array(diffs)
        if np.sum(diffs) == 0: break
        ind = np.argmax(diffs)
        keep_records.append(records[ind])
        mask = np.logical_and(mask, records[ind][1])
        del records[ind]

    print('step 5: save results ...')
    with open(os.path.join(opts.result_path, '{}_{}.data'.format(name, opts.suffix)), 'wb') as f:
        pickle.dump(keep_records, f)


if __name__ == '__main__':
    opts = get_opts()
    main(opts)
