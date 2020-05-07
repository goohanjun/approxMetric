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
    def __init__(self, data_name, size=100):
        assert data_name in ["ag_news", "imdb"]

        self.model = api.load('glove-twitter-25')
        print("Model loaded")

        self.data_name = data_name
        self.size = size
        self.train, self.test = self.load()

    def get_batch(self, batch_size=10, mode='train'):
        if mode == 'train':
            sentences, dist = self.train
        if mode == 'test':
            sentences, dist = self.test

        batch = []
        ds = len(sentences)
        for i in range(ds):
            for j in range(ds):
                batch.append((sentences[i], sentences[j], dist[i, j]))
                if len(batch) == batch_size:
                    yield batch
                    batch.clear()
        if len(batch) > 0:
            yield batch
            batch.claer()

        try:
            with open(f'./{self.data_name}_{self.size}.pkl', 'rb') as f:
                print("Load from pickle")
                train, test = pickle.load(f)
        except Exception as e:
            print("Load from pickle failed")
            sentences = get_sentences(self.data_name, size=self.size, mode='train')
            dists = compute_wmd_for_all_pairs(self.model, sentences)
            train = (sentences, dists)

            sentences = get_sentences(self.data_name, size=100, mode='test')
            dists = compute_wmd_for_all_pairs(self.model, sentences)
            test = (sentences, dists)

            with open(f'./{self.data_name}_{self.size}.pkl', 'wb') as f:
                pickle.dump((train, test), f)
        return train, test


def gensim_test():
    w2v_model = Word2Vec(common_texts, size=20, min_count=1)
    print(len(common_texts), "documents")
    index = WmdSimilarity(common_texts, w2v_model)
    query = "human user".split()
    sims = index[query]
    print(sims)


def preprocess(sentence):
    # download('stopwords')  # Download stopwords list.
    stop_words = stopwords.words('english')
    return [w for w in sentence.lower().split() if w not in stop_words]


def torchtext_ag_news(size=200, mode='train'):
    print(f"AG_NEWS {size} {mode}")

    try:
        with open(f'./ag_news_sentences_{size}_{mode}.pkl', 'rb') as f:
            sentences = pickle.load(f)

    except Exception as e:
        train_dataset, test_dataset = torchtext.datasets.text_classification.AG_NEWS(root='./AGNews',)
        voca = train_dataset.get_vocab()
        if mode == 'train':
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        elif mode == 'test':
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        sentences = []
        for i, b in enumerate(loader):
            if i == size: break
            label, sentence = b[0], b[1]
            sentence = ' '.join([voca.itos[w] for w in sentence.reshape(-1).cpu().numpy()])
            sentences.append(sentence)

        with open(f'./ag_news_sentences_{size}_{mode}.pkl', 'wb') as f:
            pickle.dump(sentences, f)

    return sentences


def torchtext_imdb(size=200, mode='train'):
    print(f"imdb {size} {mode}")
    try:
        with open(f'./imdb_sentences_{size}_{mode}.pkl', 'rb') as f:
            sentences = pickle.load(f)

    except Exception as e:
        # set up fields
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)

        # make splits for data
        train, test = datasets.IMDB.splits(TEXT, LABEL)

        # build the vocabulary
        TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
        LABEL.build_vocab(train)
        itos = TEXT.vocab.itos

        pretrained_embeddings = TEXT.vocab.vectors

        # make iterator for splits
        train_loader, test_loader = data.BucketIterator.splits((train, test), batch_size=1)

        if mode == 'train':
            loader = train_loader
        elif mode == 'test':
            loader = test_loader

        sentences = []
        for i, train_batch in enumerate(loader):
            if i == size: break
            sentence, length = train_batch.text[0], train_batch.text[1]
            sentence = ' '.join([itos[w] for w in sentence.reshape(-1).cpu().numpy()])
            sentences.append(sentence)

        with open(f'./imdb_sentences_{size}_{mode}.pkl', 'wb') as f:
            pickle.dump(sentences, f)

    return sentences


def compute_wmd_for_all_pairs(model, sentences):
    dists = np.zeros(shape=(len(sentences), len(sentences)))
    cnt_pairs = 0
    start = time.time()
    for i, sentence_1 in enumerate(sentences):
        for j, sentence_2 in enumerate(sentences):
            if i > j: continue
            cnt_pairs += 1
            # TODO: need to replace pyemd
            dist = model.wmdistance(sentence_1, sentence_2)
            dists[i, j] = dist
            dists[j, i] = dist
    end = time.time()
    print(f"Compute WMD {len(sentences)} sentences")
    print(f"Computing WMD for {cnt_pairs} pairs took {end - start:.2f} seconds")
    print(f"Avg. takes {(end - start) / cnt_pairs:.2f} seconds")
    return dists


def get_sentences(name, size=20, mode='train'):
    if name == "ag_news":
        sentences = torchtext_ag_news(size, mode)

    elif name == "imdb":
        sentences = torchtext_imdb(size, mode)

    print(f"{name} {len(sentences)} sentences. \nExample: {sentences[0]}")

    return [preprocess(s) for s in sentences]


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
