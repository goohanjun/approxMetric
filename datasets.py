from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import gensim.downloader as api

import torch
import torchtext
import torchtext.data as data
import torchtext.datasets as datasets

from nltk import download
from nltk.corpus import stopwords

import time
import pickle
import numpy as np


class DataFactory:
    def __init__(self):

        self.train, self.test_1, self.test_2 = self.load()

    def load(self):
        with open('./wmd_results_500.pkl', 'rb') as f:
            data = pickle.load(f)

        train_set = {i for i in range(400)}
        test_set = {i for i in range(400, 500)}
        train_result, test_1_result, test_2_result = {}, {}, {}
        for j in range(500):
            for i in range(j):
                if i in train_set and j in train_set:
                    train_result[(i, j)] = data[(i, j)]
                elif i in train_set and j in test_set:
                    test_1_result[(i, j)] = data[(i, j)]
                elif i in test_set and j in test_set:
                    test_2_result[(i, j)] = data[(i, j)]

        return train_result, test_1_result, test_2_result

    def get_batch(self, batch_size=10, mode='train'):
        if mode == 'train':
            sentences, dist = self.train
        if mode == 'test':
            sentences, dist = self.test




if __name__ == "__main__":

    # model = api.load('glove-twitter-25')
    # for data_name in ["ag_news", "imdb"]:
    #     sentences = get_sentences(data_name, size=10)
    #     compute_wmd_for_all_pairs(model, sentences)

    data_factory = DataFactory(data_name='ag_news', size=10)

    for b in data_factory.get_batch(batch_size=2, mode='train'):
        print(b)
        break

    data_factory.model.
