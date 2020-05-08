import torch
import torch.nn as nn
import torchtext
import torchtext.data as data
import torchtext.datasets as datasets

import random
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class DataFactory:
    def __init__(self, size=500):
        self.size = size
        self.train, self.valid, self.test_1, self.test_2 = self.load()

        self.embedding = self.load_embedding()

        self.init_performance()

    def init_performance(self):
        self.collected_approx_dists = {}

    def collect(self, mode, keys, true_dist, approx_dist):
        true_dist = true_dist.cpu().numpy()
        approx_dist = approx_dist.detach().cpu().numpy()

        for i, k in enumerate(keys):
            self.collected_approx_dists[k] = approx_dist[i]
        return

    def eval_performance(self):
        perf_dict = {}

        train_size = int(self.size * 0.8)
        wmd_dist_matrix = np.zeros(shape=(self.size, self.size))
        rwmd_dist_matrix = np.zeros(shape=(self.size, self.size))
        approx_dist_matrix = np.zeros(shape=(self.size, self.size))

        for (i, j), v  in self.results.items():
            wmd_dist_matrix[(i, j)] = v['dist_wmd']
            wmd_dist_matrix[(j, i)] = v['dist_wmd']
            rwmd_dist_matrix[(i, j)] = v['dist_rwmd']
            rwmd_dist_matrix[(j, i)] = v['dist_rwmd']

        for (i, j), v in self.collected_approx_dists.items():
            approx_dist_matrix[(i, j)] = v
            approx_dist_matrix[(j, i)] = v

        n_pairs = len(wmd_dist_matrix[:, train_size:] .nonzero()[0])
        perf_dict['test_mse_rwmd'] = np.sum((wmd_dist_matrix[:, train_size:] - rwmd_dist_matrix[:, train_size:]) ** 2.) / n_pairs
        perf_dict['test_mse_approx'] = np.sum((wmd_dist_matrix[:, train_size:] - approx_dist_matrix[:, train_size:]) ** 2.) / n_pairs
        perf_dict['test_n_pairs'] = n_pairs // 2  # symmetry matrix

        n_pairs = len(wmd_dist_matrix[:, :train_size] .nonzero()[0])
        perf_dict['train_mse_rwmd'] = np.sum((wmd_dist_matrix[:, :train_size] - rwmd_dist_matrix[:, :train_size]) ** 2.) / n_pairs
        perf_dict['train_mse_approx'] = np.sum((wmd_dist_matrix[:, :train_size] - approx_dist_matrix[:, :train_size]) ** 2.) / n_pairs
        perf_dict['train_n_pairs'] = n_pairs // 2

        def dcg_at_k(r, k, method=0):
            r = np.asfarray(r)[:k]
            if r.size:
                if method == 0:
                    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
                elif method == 1:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                else:
                    raise ValueError('method must be 0 or 1.')
            return 0.

        def ndcg_at_k(r, k=20, method=1):
            dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
            if not dcg_max:
                return 0.
            return dcg_at_k(r, k, method) / dcg_max

        def get_ndcg(true_dists, target_scores, top_k):
            merged_list = list(zip(true_dists, target_scores))

            rel_list = sorted(merged_list, key=lambda x: x[0])  # lower true distance come first
            _, rel_list = zip(*rel_list)
            ndcg = ndcg_at_k(rel_list, top_k)
            return ndcg

        # Compute ndcg
        wmd_scores = 10000. - wmd_dist_matrix
        rwmd_scores = 10000. - rwmd_dist_matrix
        approx_scores = 10000. - approx_dist_matrix

        for top_k in [10, 20, 50]:

            wmd_ndcg, rwmd_ndcg, approx_ndcg = [], [], []

            for q_idx in range(train_size, self.size):
                q_true_dists = list(wmd_dist_matrix[q_idx, :])

                q_wmd_scores = list(wmd_scores[q_idx, :])
                q_rwmd_scores = list(rwmd_scores[q_idx, :])
                q_approx_scores = list(approx_scores[q_idx, :])

                wmd_ndcg.append(get_ndcg(q_true_dists, q_wmd_scores, top_k))
                rwmd_ndcg.append(get_ndcg(q_true_dists, q_rwmd_scores, top_k))
                approx_ndcg.append(get_ndcg(q_true_dists, q_approx_scores, top_k))

            perf_dict[f'wmd_ndcg_{top_k}'] = np.mean(rwmd_ndcg)
            perf_dict[f'rwmd_ndcg_{top_k}'] = np.mean(rwmd_ndcg)
            perf_dict[f'approx_ndcg_{top_k}'] = np.mean(approx_ndcg)

        return perf_dict

    def load_embedding(self):
        # make splits for data
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)
        train, test = datasets.IMDB.splits(TEXT, LABEL)

        # build the vocabulary
        TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
        TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300))

        pretrained_embeddings = TEXT.vocab.vectors

        # assign pretrained embedding
        n_voca, n_dim = pretrained_embeddings.shape
        embedding = nn.Embedding(n_voca, n_dim)
        embedding.weight.data = pretrained_embeddings
        return embedding

    def load(self):
        try:
            with open(f'./wmd_results_{self.size}.pkl', 'rb') as f:
                results = pickle.load(f)
            with open(f'./wmd_sentences_{self.size}.pkl', 'rb') as f:
                sentences = pickle.load(f)
            print(f"wmd_results_{self.size}.pkl Loaded!")

            self.results = results
            self.sentences = sentences

        except Exception:
            print(f"wmd_results_{self.size}.pkl does not exist!")
            exit()

        # split 8:2
        train_set_size = int(self.size * 0.8)
        train_set = {i for i in range(train_set_size)}
        test_set = {i for i in range(train_set_size, self.size)}

        train_result, valid_result, test_1_result, test_2_result = {}, {}, {}, {}
        for j in range(500):
            for i in range(j):
                if i in train_set and j in train_set:
                    if random.random() > 0.2:
                        train_result[(i, j)] = results[(i, j)]
                    else:
                        valid_result[(i, j)] = results[(i, j)]
                elif i in train_set and j in test_set:
                    test_1_result[(i, j)] = results[(i, j)]
                elif i in test_set and j in test_set:
                    test_2_result[(i, j)] = results[(i, j)]

        return train_result, valid_result, test_1_result, test_2_result

    def get_batch(self, batch_size=10, mode='train'):  # train, test_1, test_2
        results_dict = {'train': self.train, 'valid': self.valid, 'test_1': self.test_1, 'test_2': self.test_2}
        results = results_dict[mode]

        sentences, recon_sentences = self.sentences

        keys, batch = [], []
        for k, v in results.items():
            i, j = k
            keys.append(k)
            batch.append((sentences[i], sentences[j], v['dist_wmd']))
            if len(batch) == batch_size:
                yield self.tensorize(batch, keys)
                batch.clear(); keys.clear()
        if len(batch) > 0:
            yield self.tensorize(batch, keys)

    def tensorize(self, batch, keys):
        sentences_1, sentences_2, dists = zip(*batch)
        lengths_1 = torch.LongTensor([len(sentence) for sentence in sentences_1])
        lengths_2 = torch.LongTensor([len(sentence) for sentence in sentences_2])
        dists = torch.FloatTensor(dists)

        max_length_1 = max([len(sentence) for sentence in sentences_1])
        max_length_2 = max([len(sentence) for sentence in sentences_2])

        padded_sentences_1, padded_sentences_2 = [], []
        for s in sentences_1:
            padded_sentences_1.append(s + [0] * (max_length_1 - len(s)))
        for s in sentences_2:
            padded_sentences_2.append(s + [0] * (max_length_2 - len(s)))

        assert len(set({len(sentence) for sentence in padded_sentences_1})) == 1
        assert len(set({len(sentence) for sentence in padded_sentences_2})) == 1

        padded_sentences_1 = torch.LongTensor(padded_sentences_1)
        padded_sentences_2 = torch.LongTensor(padded_sentences_2)

        padded_sentences_1 = self.embedding(padded_sentences_1)  # [batch, max_len_1, hidden]
        padded_sentences_2 = self.embedding(padded_sentences_2)  # [batch, max_len_2, hidden]

        if torch.cuda.is_available():
            lengths_1 = lengths_1.cuda()
            lengths_2 = lengths_2.cuda()

        padded_sentences_1 = pack_padded_sequence(padded_sentences_1, lengths_1.tolist(), batch_first=True, enforce_sorted=False)
        padded_sentences_2 = pack_padded_sequence(padded_sentences_2, lengths_2.tolist(), batch_first=True, enforce_sorted=False)

        if torch.cuda.is_available():
            padded_sentences_1 = padded_sentences_1.cuda()
            padded_sentences_2 = padded_sentences_2.cuda()
            dists = dists.cuda()

        return keys, padded_sentences_1, padded_sentences_2, dists


if __name__ == "__main__":

    data_factory = DataFactory(size=30)
    gru = nn.RNN(input_size=300, hidden_size=15, num_layers=1, bidirectional=False, batch_first=True)

    for b in data_factory.get_batch(batch_size=2, mode='train'):
        keys, sentences_1, sentences_2, dists = b
        print(sentences_1)
        print(sentences_2)
        print(dists)

        packed_output, hidden = gru(sentences_1)
        print("hidden", hidden.shape)
        break
