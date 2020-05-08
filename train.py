import torch
import argparse
from utils import Map, str2bool
import random
import numpy as np
from model import ApproxEMD
from torch.utils.tensorboard import SummaryWriter
from datasets import DataFactory


def train_single_epoch(args, model, optimizer, data_factory, summary_writer):
    pass

def train(args, model, optimizer, data_factory, summary_writer):
    for epoch in range (100):
        train_single_epoch(args, model, optimizer, data_factory, summary_writer)

        for b in data_factory.iter_batch(mode='valid'):
            data_factory.eval()
    pass


def main(args):
    # Set seed
    torch.manual_seed(0);random.seed(0);np.random.seed(0)

    # Load dataset
    # data_factory = DataFactory()

    # Load model
    model = ApproxEMD(args.data_dim, n_hidden=args.n_hidden)

    if torch.cuda.is_available(): model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-6)
    summary_writer = SummaryWriter(f'./logs/{args.model_name}')
    train(args, model, optimizer, data_factory, summary_writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_dim', type=int, default=64, help="data dimension size")
    parser.add_argument('--n_hidden', type=int, default=64, help="hidden dimension size")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")

    # Model
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--model', type=str, default='baseline')  # baseline, comm, comm_p
    parser.add_argument('--loss', type=str, default='mlse')  # mse, mlse, logsumexp
    parser.add_argument('--out_layer', type=str, default='sigmoid')  # sigmoid, exp
    parser.add_argument('--use_upper', type=str2bool, nargs='?',
                        const=True, default=False, help="True if you wanna print messages")

    # Debug
    parser.add_argument('--test', type=str2bool, nargs='?',
                        const=True, default=False, help="True if you wanna print messages")

    args = Map(vars(parser.parse_args()))
    print(args)

    args.model_name = "testEMD"

    with open('./wmd_results_500.pkl', 'rb') as f:
        data = pickle.load(f)


    main(args)
