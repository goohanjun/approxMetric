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
    def __init__(self, size=500, ratio=False, lb='rwmd'):
        self.use_cuda = False

        self.size = size
        self.ratio = ratio
        self.lb = lb
        self.train, self.valid, self.test_1, self.test_2 = self.load()

        self.embedding = self.load_embedding()
        self.init_performance()
        self.default_perf_dict = self.measure_approx_performance()

    def cuda(self):
        self.use_cuda = True

    def measure_approx_performance(self):
        perf_dict = {}
        dist_types = [k.replace('dist_', '') for k in self.results[(0, 1)].keys() if 'dist' in k]

        dist_matrix_dict = {}
        for dist_type in dist_types:
            dist_matrix = np.zeros(shape=(self.size, self.size))
            for (i, j), v in self.results.items():
                dist_matrix[j, i] = dist_matrix[i, j] = v[f'dist_{dist_type}']

            dist_perf_dict = self.eval_dist_matrix(dist_name=f"SOL_{dist_type}", dist_matrix=dist_matrix)
            perf_dict.update(dist_perf_dict)

            dist_matrix_dict[dist_type] = dist_matrix

        # Ensemble
        def ensemble_dists(dist_a, dist_b, ensemble_type='harmonic'):
            if ensemble_type == 'harmonic':
                avg = 2 * dist_a * dist_b / (dist_a + dist_b + 1e-8)
            elif ensemble_type == 'geometric':
                avg = (dist_a + dist_b) / 2.
            elif ensemble_type == 'mean':
                avg = np.sqrt(dist_a * dist_b + 1e-8)
            else:
                raise NotImplementedError
            return avg

        for lower_type in ['rwmd', 'omr', 'act', 'ict']:
            for ensemble_type in ['mean', 'geometric', 'harmonic']:
                dist_matrix = ensemble_dists(dist_matrix_dict[lower_type], dist_matrix_dict['UB_G'], ensemble_type)
                dist_perf_dict = self.eval_dist_matrix(dist_name=f"ENS_{ensemble_type}_{lower_type}", dist_matrix=dist_matrix)
                perf_dict.update(dist_perf_dict)

        return perf_dict

    def init_performance(self):
        self.collected_approx_dists = {}

    def collect(self, keys, approx_dist):
        approx_dist = approx_dist.detach().cpu().numpy()
        for i, k in enumerate(keys):
            self.collected_approx_dists[k] = approx_dist[i]
        return

    @staticmethod
    def mse_score(dist_matrix_1, dist_matrix_2, train_size, test_size):
        area_all = np.sum((dist_matrix_1 - dist_matrix_2) ** 2.)
        area_a = np.sum((dist_matrix_1[:train_size, :train_size] - dist_matrix_2[:train_size, :train_size]) ** 2.)
        area_b = np.sum((dist_matrix_1[train_size:, train_size:] - dist_matrix_2[train_size:, train_size:]) ** 2.)
        area_d = (area_all - area_a - area_b) / 2

        mse_train = area_a / (train_size * train_size)
        mse_test = (area_d + area_b) / (train_size * test_size + test_size * (test_size + 1) / 2)
        return mse_train, mse_test

    @staticmethod
    def mape_score(true_dist_matrix, approx_dist_matrix, train_size, test_size):
        # relative error
        mape = np.abs(true_dist_matrix - approx_dist_matrix) / (true_dist_matrix+1e-8)

        tmp_sum = np.sum(mape[:train_size, :train_size])
        mape_train = tmp_sum / (train_size * train_size)
        mape_test = (np.sum(mape) - tmp_sum) / ((train_size+test_size) ** 2. - train_size**2.)

        return mape_train, mape_test

    @ staticmethod
    def comp_score(dists_1, dists_2):
        # Compute triplet comparison score
        size = len(dists_1) * (len(dists_1) - 1) // 2
        comp_1 = dists_1.reshape(-1, 1) >= dists_1.reshape(1, -1)
        comp_2 = dists_2.reshape(-1, 1) >= dists_2.reshape(1, -1)
        score = (np.sum(comp_1 & comp_2) - len(dists_1)) / size
        return score

    def eval_dist_matrix(self, dist_name, dist_matrix):
        train_size = int(self.size * 0.8)
        test_size = self.size - train_size

        perf_dict = {}

        mse_train, mse_test = self.mse_score(self.wmd_dist_matrix, dist_matrix, train_size, test_size)
        perf_dict[f'{dist_name}/mse/train'] = mse_train
        perf_dict[f'{dist_name}/mse/test'] = mse_test

        mape_train, mape_test = self.mape_score(self.wmd_dist_matrix, dist_matrix, train_size, test_size)
        perf_dict[f'{dist_name}/mape/train'] = mape_train
        perf_dict[f'{dist_name}/mape/test'] = mape_test

        comp_scores = []
        for q_idx in range(self.size):
            comp_scores.append(self.comp_score(self.wmd_dist_matrix[q_idx, :], dist_matrix[q_idx, :]))

        perf_dict[f'{dist_name}/comp_accuracy/train'] = np.mean(comp_scores[:train_size])
        perf_dict[f'{dist_name}/comp_accuracy/test'] = np.mean(comp_scores[train_size:])

        return perf_dict

    def eval_performance(self):
        # load existing approximation results
        perf_dict = self.default_perf_dict.copy()

        approx_dist_matrix = np.zeros(shape=(self.size, self.size))
        for (i, j), v in self.collected_approx_dists.items():
            approx_dist_matrix[j, i] = approx_dist_matrix[i, j] = v

        # scale up, because true distance was fed with normalization
        if self.ratio:
            # r = (self.wmd_dist_matrix - self.lb_matrix) / (self.ub_matrix - self.lb_matrix + 1e-6)  [0, 1]
            approx_dist_matrix = approx_dist_matrix * (self.ub_matrix - self.lb_matrix + 1e-6) + self.lb_matrix
        else:
            approx_dist_matrix = approx_dist_matrix * self.std + self.mu

        approx_perf = self.eval_dist_matrix(dist_name='approx', dist_matrix=approx_dist_matrix)
        perf_dict.update(approx_perf)

        return perf_dict

    def load_embedding(self):
        # make splits for data
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)
        train, test = datasets.IMDB.splits(TEXT, LABEL)

        # build the vocabulary
        TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
        LABEL.build_vocab(train)
        stoi = TEXT.vocab.stoi

        sentences = []
        _, recon_sentences = self.sentences
        for recon_sentence in recon_sentences:
            sentence = [stoi[w] for w in recon_sentence.split()]
            sentences.append(sentence)
        self.sentences = (sentences, recon_sentences)

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
        train_size = int(self.size * 0.8)
        train_set = {i for i in range(train_size)}
        test_set = {i for i in range(train_size, self.size)}

        self.wmd_dist_matrix = np.zeros((self.size, self.size))
        self.lb_matrix, self.ub_matrix = np.zeros((self.size, self.size)), np.zeros((self.size, self.size))
        for (i, j), v in results.items():
            self.wmd_dist_matrix[j, i] = self.wmd_dist_matrix[i, j] = v['dist_wmd']
            self.lb_matrix[j, i] = self.lb_matrix[i, j] = v[f'dist_{self.lb}']
            self.ub_matrix[j, i] = self.ub_matrix[i, j] = v['dist_UB_G']

        self.mu = np.mean(self.wmd_dist_matrix[:train_size, :train_size])
        self.std = np.std(self.wmd_dist_matrix[:train_size, :train_size])

        normalized_wmd_dist_matrix = (self.wmd_dist_matrix - self.mu) / self.std

        r_dist_matrix = (self.wmd_dist_matrix - self.lb_matrix) / (self.ub_matrix - self.lb_matrix + 1e-6)

        if self.ratio:
            ans_matrix = r_dist_matrix
        else:
            ans_matrix = normalized_wmd_dist_matrix

        train_result, valid_result, test_1_result, test_2_result = {}, {}, {}, {}
        for j in range(self.size):
            for i in range(j):
                if i in train_set and j in train_set:
                    if random.random() > 0.10:
                        train_result[(i, j)] = ans_matrix[i, j]
                    else:
                        valid_result[(i, j)] = ans_matrix[i, j]
                elif i in train_set and j in test_set:
                    test_1_result[(i, j)] = ans_matrix[i, j]
                elif i in test_set and j in test_set:
                    test_2_result[(i, j)] = ans_matrix[i, j]

        print(f"train / valid / test_1 / test_2 =  {len(train_result)} / {len(valid_result)} / {len(test_1_result)} / {len(test_2_result)}")
        return train_result, valid_result, test_1_result, test_2_result

    def get_batch(self, batch_size=10, mode='train'):  # train, test_1, test_2
        results_dict = {'train': self.train, 'valid': self.valid, 'test_1': self.test_1, 'test_2': self.test_2}
        results = results_dict[mode]

        sentences, recon_sentences = self.sentences

        result_keys = list(results.keys())
        if mode == 'train':
            random.shuffle(result_keys)

        keys, batch = [], []
        for k in result_keys:
            v = results[k]
            i, j = k
            keys.append(k)
            batch.append((sentences[i], sentences[j], v))
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

        if self.use_cuda:
            lengths_1 = lengths_1.cuda()
            lengths_2 = lengths_2.cuda()

        padded_sentences_1 = pack_padded_sequence(padded_sentences_1, lengths_1.tolist(), batch_first=True, enforce_sorted=False)
        padded_sentences_2 = pack_padded_sequence(padded_sentences_2, lengths_2.tolist(), batch_first=True, enforce_sorted=False)

        if self.use_cuda:
            padded_sentences_1 = padded_sentences_1.cuda()
            padded_sentences_2 = padded_sentences_2.cuda()
            dists = dists.cuda()

        return keys, padded_sentences_1, padded_sentences_2, dists


if __name__ == "__main__":

    data_factory = DataFactory(size=30)
    gru = nn.GRU(input_size=300, hidden_size=15, num_layers=1, bidirectional=True, batch_first=True)

    for b in data_factory.get_batch(batch_size=3, mode='train'):
        keys, sentences_1, sentences_2, dists = b
        print(sentences_1)
        print(sentences_2)
        print(dists)

        packed_output, hidden = gru(sentences_1)
        print("hidden", hidden.shape)
        break
