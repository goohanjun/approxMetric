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
from collections import defaultdict
from utils import Map, str2bool

from distances import *


def prepare_emd(sentence_1, sentence_2, embedding):
    total_word_list = list(set(sentence_1) | set(sentence_2))
    hist_1, hist_2 = np.zeros(len(total_word_list)), np.zeros(len(total_word_list))

    idx_map = {w: idx for idx, w in enumerate(total_word_list)}
    for w in sentence_1: hist_1[idx_map[w]] += 1.
    for w in sentence_2: hist_2[idx_map[w]] += 1.
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    # dists = np.zeros((len(total_word_list), len(total_word_list)))

    word_tensor = torch.LongTensor(total_word_list)
    if torch.cuda.is_available():
        word_tensor.cuda()

    word_emb_tensor = embedding(word_tensor)
    dists = torch.sqrt(torch.sum((word_emb_tensor.unsqueeze(0) - word_emb_tensor.unsqueeze(1)) ** 2., dim=-1) + 1e-10)
    dists = dists.detach().cpu().numpy().astype(np.float64)

    return hist_1, hist_2, dists


def compute_wmd(hist_1, hist_2, dists):
    dist = emd(hist_1, hist_2, dists)
    return dist


def compute_emd_with_flow(hist_1, hist_2, dists):
    dist, flow = emd_with_flow(hist_1, hist_2, dists)
    return dist, flow


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k - 1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def print_flow(flow, sentence_1, sentence_2, itos, k=10):
    total_word_list = list(set(sentence_1) | set(sentence_2))
    total_word_list_recon = [itos[w_idx] for w_idx in total_word_list]
    print(f"top-{k} flows")
    for ii, jj in k_largest_index_argsort(np.asarray(flow), k):
        print(f"flow: {flow[ii][jj]:.2f}\t{total_word_list_recon[ii]: <10} -> {total_word_list_recon[jj]}")


def load_emb():
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)

    itos, stoi = TEXT.vocab.itos, TEXT.vocab.stoi

    stop_words = stopwords.words('english')
    stop_words_idx = [TEXT.vocab.stoi[w] for w in stop_words]

    pretrained_embeddings = TEXT.vocab.vectors

    # assign pretrained embedding
    n_voca, n_dim = pretrained_embeddings.shape
    embedding = nn.Embedding(n_voca, n_dim)
    if torch.cuda.is_available():
        embedding.weight.data = pretrained_embeddings.cuda()
    else:
        embedding.weight.data = pretrained_embeddings
    return embedding, itos, stoi, stop_words_idx, train, test


def load(args):
    size = args.data_size
    try:
        with open(f"./wmd_sentences_{size}.pkl", 'rb') as f:
            idx_sentences, recon_sentences = pickle.load(f)
            sentences = (idx_sentences, recon_sentences)

        with open(f"./wmd_results_{size}.pkl", 'rb') as f:
            results = pickle.load(f)

    except (OSError, IOError) as e:
        results = {}
        sentences = []
    return results, sentences


def label_dists(sentences, embedding, results_prev):
    funcs = [compute_wmd, compute_greenkhorn_0_1, compute_greenkhorn_0_5,  # NIPS'17
             compute_rwmd, compute_omr, compute_act, compute_ict,  # ICML'19
             compute_UB_G,]  # VLDB'13
#             compute_hmean_rwmd_UB_G, compute_hmean_ict_UB_G]  # TKDE'19

    n_pairs, results = 0, defaultdict(dict)
    results.update(results_prev)

    total_start = time.time()
    for i, sentence_1 in tqdm(enumerate(sentences), total=len(sentences)):
        for j, sentence_2 in enumerate(sentences):
            if j > i:
                n_pairs += 1

                hist_1, hist_2, dists = prepare_emd(sentence_1, sentence_2, embedding)

                for func in funcs:
                    f_name = func.__name__.replace('compute_', '')

                    if f'dist_{f_name}' in results[(i, j)]: continue

                    tic = time.time()
                    dist = func(hist_1, hist_2, dists)
                    toc = time.time()

                    results[(i, j)][f'dist_{f_name}'] = dist
                    results[(i, j)][f'time_{f_name}'] = toc - tic

                if n_pairs < 2: print(f'Results example\n{results[(i, j)]}')

                # Assertion
                eps = 1e-5
                assert results[(i, j)]['dist_wmd'] <= results[(i, j)]['dist_UB_G'] + eps, "upper" + str(results[(i, j)])
                assert results[(i, j)]['dist_rwmd'] <= results[(i, j)]['dist_omr'] + eps, "omr" + str(results[(i, j)])
                assert results[(i, j)]['dist_omr'] <= results[(i, j)]['dist_act'] + eps, "act" + str(results[(i, j)])
                assert results[(i, j)]['dist_act'] <= results[(i, j)]['dist_ict'] + eps, "ict" + str(results[(i, j)])
                assert results[(i, j)]['dist_ict'] <= results[(i, j)]['dist_wmd'] + eps, "wmd" + str(results[(i, j)])

    total_end = time.time()
    print(f"Total time took {total_end - total_start:.2f} seconds. Avg. = {(total_end - total_start) / n_pairs:.4f}")

    return results


def gen(args):
    embedding, itos, stoi, stop_words_idx, train, test,  = load_emb()
    print("Embedding loaded")

    results_prev, sentences = load(args)

    if len(results_prev) > 0:
        # Loaded
        _, recon_sentences = sentences
        idx_sentences = []
        for recon_sentence in recon_sentences:
            idx_sentences.append([stoi[w] for w in recon_sentence.split() if stoi[w] not in stop_words_idx])
    else:
        # Generate
        size = args.data_size
        train_loader, test_loader = data.BucketIterator.splits((train, test), batch_size=1)
        idx_sentences, recon_sentences, labels = [], [], []
        for i, train_batch in enumerate(train_loader):
            if i == size: break
            idx_sentence, length, label = train_batch.text[0], train_batch.text[1], train_batch.label
            length = length.view(-1).cpu().numpy()[0]
            idx_sentence = idx_sentence.view(-1).cpu().numpy()
            idx_sentence = [w_idx for w_idx in idx_sentence if w_idx not in stop_words_idx]
            recon_sentence = ' '.join([itos[w_idx] for w_idx in idx_sentence])

            if i == 0: print(f"Sentence Example\nlength = {length}\nidx_sentence: {idx_sentence}\nrecon_sentence: {recon_sentence}")

            idx_sentences.append(idx_sentence)
            recon_sentences.append(recon_sentence)
            labels.append(label.item())
    print(f"{len(idx_sentences)} sentences loaded")

    results = label_dists(idx_sentences, embedding, results_prev)

    with open(f"./wmd_sentences_{len(idx_sentences)}.pkl", 'wb') as f:
        pickle.dump((idx_sentences, recon_sentences), f)

    with open(f"./wmd_results_{len(idx_sentences)}.pkl", 'wb') as f:
        pickle.dump(results, f)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=100, help="number of sentence size")
    args = Map(vars(parser.parse_args()))
    print(args)

    gen(args)
