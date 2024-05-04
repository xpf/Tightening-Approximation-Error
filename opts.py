import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='/data/xpf/datasets')
    parser.add_argument('--weight_path', type=str, default='/data/xpf/weights')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--evaluation_path', type=str, default='./evaluations')

    parser.add_argument('--data_name', type=str, default='cifar100')
    parser.add_argument('--model_name', type=str, default='Cui2020Learnable_34_10_LBGAT0')
    parser.add_argument('--n_class', type=int, default=100)
    parser.add_argument('--norm', type=str, default='linf')

    parser.add_argument('--data_num', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--target', type=int, default=-1)

    parser.add_argument('--n_pop', type=int, default=100)
    parser.add_argument('--n_gen', type=int, default=50)
    parser.add_argument('--cx_pb', type=float, default=0.5)
    parser.add_argument('--mut_pb', type=float, default=0.3)

    parser.add_argument('--space', type=int, default=0)
    parser.add_argument('--suffix', type=int, default=0)
    parser.add_argument('--baseline', type=int, default=2)

    parser.add_argument('--rname', type=str)
    parser.add_argument('--rid', type=int)

    opts = parser.parse_args()
    return opts
    # 早停技术
    # 多损失评估
    # 部分数据评估
    # 多随机启动评估
