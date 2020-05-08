import torch
import torch.nn as nn
import torchtext
import torchtext.data as data
import torchtext.datasets as datasets

from nltk.corpus import stopwords
from nltk import download

import numpy as np
from pyemd import emd
from pyemd import emd_with_flow

import argparse
import pickle
from tqdm import tqdm
import time
from utils import Map, str2bool


def prepare_emd(sentence_1, sentence_2, embedding):
    total_word_list = list(set(sentence_1) | set(sentence_2))
    hist_1, hist_2 = np.zeros(len(total_word_list)), np.zeros(len(total_word_list))

    idx_map = {w: idx for idx, w in enumerate(total_word_list)}
    for w in sentence_1: hist_1[idx_map[w]] += 1.
    for w in sentence_2: hist_2[idx_map[w]] += 1.
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    dists = np.zeros((len(total_word_list), len(total_word_list)))

    word_tensor = torch.LongTensor(total_word_list).cuda()
    word_emb_tensor = embedding(word_tensor)
    dists = torch.sqrt(torch.sum((word_emb_tensor.unsqueeze(0) - word_emb_tensor.unsqueeze(1)) ** 2., dim=-1) + 1e-10)
    dists = dists.detach().cpu().numpy().astype(np.float64)

    return hist_1, hist_2, dists


def compute_emd(sentence_1, sentence_2, embedding):
    hist_1, hist_2, dists = prepare_emd(sentence_1, sentence_2, embedding)
    dist = emd(hist_1, hist_2, dists)
    return dist


def compute_emd_with_flow(sentence_1, sentence_2, embedding):
    hist_1, hist_2, dists = prepare_emd(sentence_1, sentence_2, embedding)
    dist, flow = emd_with_flow(hist_1, hist_2, dists)
    return dist, flow


def relaxed_emd(sentence_1, sentence_2, embedding):
    total_word_list_1 = list(set(sentence_1))
    total_word_list_2 = list(set(sentence_2))

    hist_1 = np.zeros(len(total_word_list_1))
    hist_2 = np.zeros(len(total_word_list_2))
    idx_map_1 = {w: idx for idx, w in enumerate(total_word_list_1)}
    idx_map_2 = {w: idx for idx, w in enumerate(total_word_list_2)}
    for w in sentence_1: hist_1[idx_map_1[w]] += 1.
    for w in sentence_2: hist_2[idx_map_2[w]] += 1.
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    word_1_tensor = torch.LongTensor(total_word_list_1).cuda()
    word_2_tensor = torch.LongTensor(total_word_list_2).cuda()
    word_emb_tensor_1 = embedding(word_1_tensor)  # [n, 300]
    word_emb_tensor_2 = embedding(word_2_tensor)  # [m, 300]

    # [n, m, 300] -> [n, m]
    dists = torch.sqrt(
        torch.sum((word_emb_tensor_1.unsqueeze(1) - word_emb_tensor_2.unsqueeze(0)) ** 2., dim=-1) + 1e-10)

    dist_1 = torch.sum(torch.FloatTensor(hist_1).cuda() * torch.min(dists, dim=1)[0]).detach().cpu().item()
    dist_2 = torch.sum(torch.FloatTensor(hist_2).cuda() * torch.min(dists, dim=0)[0]).detach().cpu().item()
    return max(dist_1, dist_2)


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k - 1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def print_flow(flow, sentence_1, sentence_2, k=10):
    total_word_list = list(set(sentence_1) | set(sentence_2))
    total_word_list_recon = [TEXT.vocab.itos[w_idx] for w_idx in total_word_list]
    print(f"top-{k} flows")
    for ii, jj in k_largest_index_argsort(np.asarray(flow), k):
        print(f"flow: {flow[ii][jj]:.2f}\t{total_word_list_recon[ii]: <10} -> {total_word_list_recon[jj]}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=100, help="number of sentence size")
    parser.add_argument('--dump_sentence_only', type=str2bool, nargs='?',
                        const=True, default=False, help="True if you wanna print messages")
    args = Map(vars(parser.parse_args()))
    print(args)

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)
    itos = TEXT.vocab.itos

    stop_words = stopwords.words('english')
    stop_words_idx = [TEXT.vocab.stoi[w] for w in stop_words]

    pretrained_embeddings = TEXT.vocab.vectors

    # assign pretrained embedding
    n_voca, n_dim = pretrained_embeddings.shape
    embedding = nn.Embedding(n_voca, n_dim)
    embedding.weight.data = pretrained_embeddings.cuda()

    size = args.data_size
    train_loader, test_loader = data.BucketIterator.splits((train, test), batch_size=1)
    sentences, recon_sentences, labels = [], [], []
    for i, train_batch in enumerate(train_loader):
        if i == size: break
        sentence, length, label = train_batch.text[0], train_batch.text[1], train_batch.label
        length = length.reshape(-1).cpu().numpy()[0]
        sentence = sentence.reshape(-1).cpu().numpy()
        if i == 0: print(length, sentence)
        sentence = [w_idx for w_idx in sentence if w_idx not in stop_words_idx]

        sentences.append(sentence)
        recon_sentences.append(' '.join([itos[w_idx] for w_idx in sentence]))
        labels.append(label.item())

    print(f"{len(sentences)} sentences loaded")

    with open(f"./wmd_sentences_{len(sentences)}.pkl", 'wb') as f:
        pickle.dump((sentences, recon_sentences), f)

    if args.dump_sentence_only:
        exit()

    calc_flow = False

    n_pairs, results = 0, {}
    total_start = time.time()
    for i, sentence_1 in tqdm(enumerate(sentences), total=len(sentences)):
        for j, sentence_2 in enumerate(sentences):
            if j > i:
                n_pairs += 1
                tic = time.time()
                dist_wmd = compute_emd(sentence_1, sentence_2, embedding)
                tac = time.time()
                dist_rwmd = relaxed_emd(sentence_1, sentence_2, embedding)
                toc = time.time()

                results[(i, j)] = {'dist_wmd': dist_wmd, 'dist_rwmd': dist_rwmd,
                                   'time_wmd': tac - tic, 'time_rwmd': toc - tac}

                if calc_flow:
                    dist, flow = compute_emd_with_flow(sentence_1, sentence_2, embedding)
                    if n_pairs < 2:
                        print_flow(flow, sentence_1, sentence_2, k=10)
    total_end = time.time()

    print(f"Total time took {total_end - total_start:.2f} seconds. Avg. = {(total_end - total_start) / n_pairs:.4f}")

    with open(f"./wmd_results_{len(sentences)}.pkl", 'wb') as f:
        pickle.dump(results, f)
