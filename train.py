import torch
import argparse
from utils import Map, str2bool
import random
import numpy as np
from model import ApproxEMD, ApproxEMDAttention
from torch.utils.tensorboard import SummaryWriter
from datasets import DataFactory
from tqdm import tqdm


def train_single_epoch(args, model, optimizer, data_factory, mode):
    loss_func = torch.nn.MSELoss()
    n_batch, cum_loss = 0, 0.
    for b in data_factory.get_batch(batch_size=args.batch_size, mode=mode):
        n_batch += 1
        keys, sentences_1, sentences_2, dists = b

        if mode == 'train':
            model.zero_grad()

        approx_dists = model.forward(sentences_1, sentences_2)

        # for evaluation
        data_factory.collect(keys, approx_dists)

        loss = loss_func(dists, approx_dists)
        cum_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()
    return cum_loss / n_batch


def train(args, model, optimizer, data_factory):
    summary_writer_dict = {}
    algs = {k.split('/')[0] for k in data_factory.default_perf_dict.keys()}

    for alg in algs:
        summary_writer_dict[alg] = SummaryWriter(f'./logs/EMD{args.data_size}_{alg}')
    summary_writer_dict['approx'] = SummaryWriter(f'./logs/EMD{args.data_size}_{args.model_name}')

    for epoch in tqdm(range(300)):
        data_factory.init_performance()

        for mode in ['train', 'valid', 'test_1', 'test_2']:
            loss = train_single_epoch(args, model, optimizer, data_factory, mode)
            summary_writer_dict['approx'].add_scalar(f'loss/{mode}', loss, epoch)

        perf_dict = data_factory.eval_performance()
        for k, v in perf_dict.items():
            alg, key = k.split('/')
            summary_writer_dict[alg].add_scalar(f'{key}', v, epoch)

        print(f"Epoch @ {epoch}", perf_dict)


def main(args):
    # Set seed
    torch.manual_seed(0);random.seed(0);np.random.seed(0)

    # Load dataset
    data_factory = DataFactory(size=args.data_size)

    # Load model
    if 'att' in args.model_name:
        model = ApproxEMDAttention(n_hidden=args.n_hidden)
    else:
        model = ApproxEMD(n_hidden=args.n_hidden)
    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_reg)

    train(args, model, optimizer, data_factory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_size', type=int, default=30, help="Data sentence size")
    parser.add_argument('--n_hidden', type=int, default=64, help="hidden dimension size")  # 128?
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")

    # Model
    parser.add_argument('--model_name', type=str, default="attEMD", help="model name")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--l2_reg', type=float, default=1e-5, help="learning rate")

    # Debug
    parser.add_argument('--test', type=str2bool, nargs='?',
                        const=True, default=False, help="True if you wanna print messages")

    args = Map(vars(parser.parse_args()))
    args.model_name = f"{args.model_name}_{args.data_size}_nh{args.n_hidden}_reg{args.l2_reg}"
    print(args)

    main(args)
