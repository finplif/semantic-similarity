import re
import gensim
import logging
import nltk.data
import pprint
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim import corpora
from pymorphy2 import MorphAnalyzer
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %matplotlib inline
import random

import warnings
warnings.filterwarnings('ignore')

pp = pprint.PrettyPrinter(indent=4)



class semansim:
    def __init__(self, text):
        self.text = text
        self.processed_text, self.lemmd_text, self.poses, self.file, self.data = self.lemmatize(self.text)

    def lemmatize(self, new_text):
        morph = MorphAnalyzer()
        words = []
        # lemmd_sent = []
        lemmd_text = []
        poses = []  # we need this for the last part

        with open(new_text, encoding='utf-8') as f:
            text = f.read()
        text = re.sub(r'\n', ' ', text)
        text = sent_tokenize(text)

        for sent in text:
            sentences = word_tokenize(sent)
            token = [word.lower() for word in sentences]
            tokens = [word for word in token if word.isalpha()]
            words.append(tokens)

        for group in words:
            lemmd_sent = []
            for unit in group:
                ana = morph.parse(unit)
                lemma = ana[0].normal_form
                lemmd_sent.append(lemma)
                pos = ana[0]
                if (pos.tag.POS == 'PREP') or (pos.tag.POS == 'PRCL') or (pos.tag.POS == 'CONJ'):
                    poses.append(pos.normal_form)
            lemmd_sents = ' '.join(lemmd_sent)
            lemmd_text.append(lemmd_sents)
        new_text = '\n'.join(lemmd_text)
        with open('new-text.txt', 'w', encoding='utf-8') as f:
            f.write(new_text)

        f = 'new-text.txt'
        data = gensim.models.word2vec.LineSentence(f)
        return (text, lemmd_text, poses, f, data)


    def modeltrain(self, size=300, window=5, min_count=5, iter=50):
        self.model = gensim.models.Word2Vec(self.data, size=size, window=window, min_count=min_count, iter=iter)
        self.model.init_sims(replace=True)
        self.model_path = 'model.bin'
        self.model.wv.save_word2vec_format(self.model_path, binary=True)


    def get_num_words(self):
        return len(self.model.wv.vocab)


    def get_words(self):
        self.output_text = 'new_text_words.txt'
        with open(self.output_text, 'w', encoding='utf-8') as f:
            for w in sorted(self.model.wv.vocab):
                f.write("%s %s" % (w, '\n'))


    def get_n_most_similar(self, word, n=5):
        return self.model.wv.most_similar(word, topn=n)


    def get_semantic_proportion(self, positives, negatives):
        return self.model.wv.most_similar(positive=positives, negative=negatives)[0][0]


    def odd_one_out(self, line):
        return self.model.wv.doesnt_match(line.split())


    def visualize_vector_similarity(self, words):
        X = self.model[words]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        plt.scatter(coords[:, 0], coords[:, 1], color='Aquamarine')
        plt.title('Words')
        for i, word in enumerate(words):
            plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
        plt.show()


    def words_swap(self, input_sentence=''):
        new_sent = []
        if not input_sentence:
            i = random.randint(1, len(self.text))
            input_sentence = self.lemmd_text[i]
        else:
            input_sentence = self.lemmatize(input_sentence)

        trial_sent = input_sentence.split()
        for word in trial_sent:
            if (word in poses) or (word not in self.model.wv.vocab):
                new_sent.append(word)
            else:
                new_sent.append(self.get_n_most_similar(word, n=1))

        sent = ' '.join(new_sent)
        return sent


# converting a text into a lemmatized text

# def lemmatize(new_text):
#     morph = MorphAnalyzer()
#     words = []
#     # lemmd_sent = []
#     lemmd_text = []
#     poses = []  # we need this for the last part
#
#     with open(new_text, encoding='utf-8') as f:
#         text = f.read()
#     text = re.sub(r'\n', ' ', text)
#     text = sent_tokenize(text)
#
#     for sent in text:
#         sentences = word_tokenize(sent)
#         token = [word.lower() for word in sentences]
#         tokens = [word for word in token if word.isalpha()]
#         words.append(tokens)
#
#     for group in words:
#         lemmd_sent = []
#         for unit in group:
#             ana = morph.parse(unit)
#             lemma = ana[0].normal_form
#             lemmd_sent.append(lemma)
#             pos = ana[0]
#             if (pos.tag.POS == 'PREP') or (pos.tag.POS == 'PRCL') or (pos.tag.POS == 'CONJ'):
#                 poses.append(pos.normal_form)
#         lemmd_sents = ' '.join(lemmd_sent)
#         lemmd_text.append(lemmd_sents)
#     new_text = '\n'.join(lemmd_text)
#     with open('new-text.txt', 'w', encoding='utf-8') as f:
#         f.write(new_text)
#
#     f = 'new-text.txt'
#     data = gensim.models.word2vec.LineSentence(f)
#     return (text, lemmd_text, poses, f, data)

# text, lemmd_text, poses, f, data = lemmatize('gambler.txt')




# model is learning new vectors with pointed parameters
#
# model_gambler = gensim.models.Word2Vec(data, size=300, window=5, min_count=5, iter=50)
# model_gambler.init_sims(replace=True)
# model_path = 'gambler.bin'
# model_gambler.wv.save_word2vec_format(model_path, binary=True)
#
# print('total words amount - ', len(model_gambler.wv.vocab))

# with open('new-gambler-words.txt', 'w', encoding='utf-8') as f:
#     for w in sorted(model_gambler.wv.vocab):
#         f.write("%s %s" % (w, '\n'))


# the most similar words for:
# pp.pprint(model_gambler.wv.most_similar("occasion", topn=5))
# pp.pprint(model_gambler.wv.most_similar("quarrel", topn=5))
# pp.pprint(model_gambler.wv.most_similar("proud", topn=5))
# pp.pprint(model_gambler.wv.most_similar("rose", topn=5))
# pp.pprint(model_gambler.wv.most_similar("slave", topn=5))
# print()
# print(model_gambler.wv.most_similar(positive=['rose', 'madame'], negative=['morning'])[0][0])  # semantic proportion
# print(model_gambler.wv.doesnt_match("party madame morning next".split()))  # the odd one in the list


# visualization
# words = ['window', 'wheel', 'view', 'room', 'sound', 'result', 'moscow',
#          'lips', 'nature', 'clock', 'corridor', 'crowd']
# X = model_gambler[words]
# pca = PCA(n_components=2)
# coords = pca.fit_transform(X)
# plt.scatter(coords[:, 0], coords[:, 1], color='Aquamarine')
# plt.title('Words')
# for i, word in enumerate(words):
#     plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
# plt.show()

# change every word (except for function words) for the most similar ones.

# i = random.randint(1, len(text))
#
# new_sent = []
# trial_sent = lemmd_text[i]
# trial_sent = trial_sent.split()
# show_trial = ' '.join(trial_sent)
# print('take a sentence: ', show_trial)
#
# for word in trial_sent:
#     if (word in poses) or (word not in model_gambler.wv.vocab):
#         new_sent.append(word)
#     else:
#         new_sent.append(model_gambler.wv.most_similar(word, topn=1)[0][0])
#
# sent = ' '.join(new_sent)
# print('new sentence: ', sent)






