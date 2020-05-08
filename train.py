import torch
import argparse
from utils import Map, str2bool
import random
import numpy as np
from model import ApproxEMD
from torch.utils.tensorboard import SummaryWriter
from datasets import DataFactory
from tqdm import tqdm


def train_single_epoch(args, model, optimizer, data_factory, mode):
    cum_loss = 0.
    loss_func = torch.nn.MSELoss()

    for b in data_factory.get_batch(batch_size=args.batch_size, mode=mode):
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
    return cum_loss


def train(args, model, optimizer, data_factory, summary_writers):
    summary_writer_train, summary_writer_test = summary_writers
    for epoch in tqdm(range(300)):
        data_factory.init_performance()

        for mode in ['train', 'valid', 'test_1', 'test_2']:
            loss = train_single_epoch(args, model, optimizer, data_factory, mode)
            if mode == 'train': summary_writer_train.add_scalar(f'loss/loss', loss, epoch)
            elif mode == 'valid': summary_writer_test.add_scalar(f'loss/loss', loss, epoch)
            elif 'test' in mode: summary_writer_test.add_scalar(f'loss/{mode}_loss', loss, epoch)

        perf_dict = data_factory.eval_performance()
        for k, v in perf_dict.items():
            alg, key = k.split('/')
            if alg == 'approx':
                summary_writer_train.add_scalar(f'{key}', v, epoch)
            else:
                summary_writer_test.add_scalar(f'{key}', v, epoch)
        print(perf_dict)


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

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_reg)
    sw_train = SummaryWriter(f'./logs/{args.model_name}_train')
    sw_test = SummaryWriter(f'./logs/{args.model_name}_test')

    train(args, model, optimizer, data_factory, (sw_train, sw_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_size', type=int, default=30, help="Data sentence size")
    parser.add_argument('--n_hidden', type=int, default=128, help="hidden dimension size")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")

    # Model
    parser.add_argument('--model_name', type=str, default="EMD", help="model name")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--l2_reg', type=float, default=1e-5, help="learning rate")

    # Debug
    parser.add_argument('--test', type=str2bool, nargs='?',
                        const=True, default=False, help="True if you wanna print messages")

    args = Map(vars(parser.parse_args()))
    args.model_name = f"{args.model_name}_{args.data_size}_nh{args.n_hidden}_reg{args.l2_reg}"
    print(args)

    main(args)
