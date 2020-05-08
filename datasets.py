import torch
import torch.nn as nn
import torchtext
import torchtext.data as data
import torchtext.datasets as datasets

import random
import pickle
import numpy as np


class DataFactory:
    def __init__(self, size=500):
        self.size = size
        self.sentences, self.train, self.valid, self.test_1, self.test_2 = self.load()

        self.cuda = torch.cuda.is_available()
        self.embedding = self.load_embedding()
        if self.cuda: self.embedding.cuda()

        self.init_performance()

    def init_performance(self):
        self.collected_true_dists = {'train': {}, 'valid': {}, 'test_1': {}, 'test_2': {}}
        self.collected_approx_dists = {'train': {}, 'valid': {}, 'test_1': {}, 'test_2': {}}

    def collect(self, mode, keys, true_dist, approx_dist):
        true_dist = true_dist.cpu().numpy()
        approx_dist = approx_dist.detach().cpu().numpy()

        true_dist_dict = self.collected_true_dists[mode]
        approx_dist_dict = self.collected_approx_dists[mode]

        for i, k in enumerate(keys):
            true_dist_dict[k] = true_dist[i]
            approx_dist_dict[k] = approx_dist[i]
        return

    def eval_performance(self, mode):
        true_dist_dict = self.collected_true_dists[mode]
        approx_dist_dict = self.collected_approx_dists[mode]

        n_pairs, n_hits = 0, 0

        # MSE
        mse = 0.
        for k, v in true_dist_dict.items():
            mse += (v - approx_dist_dict[k])**2.

        # for i in range(n_sentence):
        #     for j in range(i + 1, n_sentence):
        #         for k in range(j + 1, n_sentence):
        #             if i == j or j == k: continue
        #             dist_1, dist_2 = true_dists[(i, j)], true_dists[(j, k)]
        #             approx_dist_1, approx_dist_2 = approx_dists[(i, j)], approx_dists[(j, k)]
        #
        #             n_pairs += 1
        #             if dist_1 > dist_2 and approx_dist_1 > approx_dist_2:
        #                 n_hits += 1
        #             elif dist_1 < dist_2 and approx_dist_1 < approx_dist_2:
        #                 n_hits += 1

        perf_dict = {"mse":mse, "n_hits": n_hits, "n_pairs": n_pairs, "comparison_accuracy": n_hits / n_pairs}
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
        if self.cuda:
            embedding.weight.data = pretrained_embeddings.cuda()
        else:
            embedding.weight.data = pretrained_embeddings
        return embedding

    def load(self):
        try:
            with open(f'./wmd_results_{self.size}.pkl', 'rb') as f:
                results = pickle.load(f)
            with open(f'./wmd_sentences_{self.size}.pkl', 'rb') as f:
                sentences = pickle.load(f)

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

        return sentences, train_result, valid_result, test_1_result, test_2_result

    def get_batch(self, batch_size=10, mode='train'):  # train, test_1, test_2
        if mode == 'train':
            results = self.train
        if mode == 'valid':
            results = self.valid
        elif mode == 'test_1':
            results = self.test_1
        elif mode == 'test_2':
            results = self.test_2

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

        if self.cuda:
            lengths_1.cuda();lengths_2.cuda();
            padded_sentences_1.cuda();padded_sentences_2.cuda();
            dists.cuda()

        padded_sentences_1 = self.embedding(padded_sentences_1)  # [batch, max_len_1, hidden]
        padded_sentences_2 = self.embedding(padded_sentences_2)  # [batch, max_len_2, hidden]

        return keys, padded_sentences_1, padded_sentences_2, lengths_1, lengths_2, dists


if __name__ == "__main__":

    # model = api.load('glove-twitter-25')
    # for data_name in ["ag_news", "imdb"]:
    #     sentences = get_sentences(data_name, size=10)
    #     compute_wmd_for_all_pairs(model, sentences)

    from torch.nn.utils.rnn import pack_padded_sequence

    data_factory = DataFactory(size=30)
    gru = nn.RNN(input_size=300, hidden_size=15, num_layers=1, bidirectional=False, batch_first=True)

    for b in data_factory.get_batch(batch_size=2, mode='train'):
        keys, sentences_1, sentences_2, lengths_1, lengths_2, dists = b
        print(sentences_1)
        print(lengths_1)
        print(sentences_2)
        print(lengths_2)
        print(dists)

        packed_input = pack_padded_sequence(sentences_2, lengths_2.tolist(), batch_first=True)
        packed_output, hidden = gru(packed_input)
        # print(packed_input)
        print("hidden", hidden.shape)

        break
