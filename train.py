import torch
import argparse
from utils import Map, str2bool
import random
import numpy as np
from model import ApproxEMD
from torch.utils.tensorboard import SummaryWriter
from datasets import DataFactory
from tqdm import tqdm


def train_single_epoch(args, model, optimizer, data_factory, summary_writer, mode):
    cum_loss = 0.
    loss_func = torch.nn.MSELoss()

    data_factory.init_performance()
    for b in data_factory.get_batch(batch_size=args.batch_size, mode=mode):
        keys, sentences_1, sentences_2, dists = b

        if mode == 'train':
            model.zero_grad()

        approx_dists = model.forward(sentences_1, sentences_2)

        # for evaluation
        data_factory.collect(mode, keys, dists, approx_dists)

        loss = loss_func(dists, approx_dists)
        cum_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()
    return cum_loss


def train(args, model, optimizer, data_factory, summary_writer):
    modes = ['train', 'valid', 'test_1', 'test_2']

    for epoch in tqdm(range(100)):

        for mode in modes:
            loss = train_single_epoch(args, model, optimizer, data_factory, summary_writer, mode)
            perf_dict = data_factory.eval_performance(mode)

            if mode == 'train': print(perf_dict)

            summary_writer.add_scalar(f'{mode}/loss', loss, epoch)
            for k, v in perf_dict.items():
                summary_writer.add_scalar(f'{mode}/{k}', v, epoch)


def main(args):
    # Set seed
    torch.manual_seed(0);random.seed(0);np.random.seed(0)

    # Load dataset
    data_factory = DataFactory(size=args.data_size)

    # Load model
    model = ApproxEMD(n_hidden=args.n_hidden)
    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    summary_writer = SummaryWriter(f'./logs/{args.model_name}')

    train(args, model, optimizer, data_factory, summary_writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_size', type=int, default=30, help="Data sentence size")
    parser.add_argument('--n_hidden', type=int, default=64, help="hidden dimension size")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")

    # Model
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

    # Debug
    parser.add_argument('--test', type=str2bool, nargs='?',
                        const=True, default=False, help="True if you wanna print messages")

    args = Map(vars(parser.parse_args()))
    args.model_name = "testEMD"
    print(args)

    main(args)
