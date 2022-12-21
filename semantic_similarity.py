import pprint
import random
import re
import warnings
import gensim
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

wnl = WordNetLemmatizer()

warnings.filterwarnings('ignore')

pp = pprint.PrettyPrinter(indent=4)


class SemanSim:
    def __init__(self, text):
        self.text = text
        self.processed_text, self.lemmd_text, self.file, self.data = self.lemmatize(self.text)
        self.model = self.modeltrain()
        self.model_keys = self.get_words()

    def lemmatize(self, new_text):
        words = []
        lemmd_text = []

        with open(new_text, encoding='utf-8') as f:
            text = f.read()
        text = re.sub(r'\n', ' ', text)
        text = sent_tokenize(text)
        # sw = nltk.corpus.stopwords.words('english')  # getting rid of stopwords
        for sent in text:
            sentences = word_tokenize(sent)
            token = [word.lower() for word in sentences]
            tokens = [word for word in token if word.isalpha()]
            # filtered = [word for word in tokens if word not in sw]
            # words.append(filtered)
            words.append(tokens)

        for group in words:
            lemmd_sent = []
            pos_tags = nltk.pos_tag(group)

            for duo in pos_tags:
                if duo[1].startswith("NN"):
                    lemma = wnl.lemmatize(duo[0], pos='n')
                elif duo[1].startswith('VB'):
                    lemma = wnl.lemmatize(duo[0], pos='v')
                elif duo[1].startswith('JJ'):
                    lemma = wnl.lemmatize(duo[0], pos='a')
                else:
                    lemma = duo[0]
                lemmd_sent.append(lemma)
            lemmd_sents = ' '.join(lemmd_sent)
            lemmd_text.append(lemmd_sents)
        new_text = '\n'.join(lemmd_text)
        with open('new-text.txt', 'w', encoding='utf-8') as f:
            f.write(new_text)

        f = 'new-text.txt'
        data = gensim.models.word2vec.LineSentence(f)
        return (text, lemmd_text, f, data)

    def modeltrain(self, vector_size=500, window=5, min_count=5,epochs=50):  # model is learning new vectors with pointed parameters
        model = gensim.models.Word2Vec(self.data, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
        return model

    def get_num_words(self):
        return len(self.model.wv)

    def get_words(self):
        output_text = 'new-text-words.txt'
        model_keys = self.model.wv.key_to_index
        with open(output_text, 'w', encoding='utf-8') as f:
            for w in sorted(model_keys):
                f.write("%s %s" % (w, '\n'))
        return model_keys

    def get_n_most_similar(self, word, n=5):  # get n most similar words
        return self.model.wv.most_similar(word, topn=n)

    def get_semantic_proportion(self, positives, negatives):
        return self.model.wv.most_similar(positive=positives, negative=negatives)[0][0]

    def odd_one_out(self, line):
        return self.model.wv.doesnt_match(line.split())

    def visualize_vector_similarity(self, list_of_words):  # visualization of similar words by their vectors
        X = self.model.wv[list_of_words]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        plt.scatter(coords[:, 0], coords[:, 1], color='Aquamarine')
        plt.title('Words')
        for i, word in enumerate(words):
            plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
        plt.show()

    def words_swap(self, input_sentence=''):  # we swap each word in the original text with the most similar one
        new_sent = []
        if not input_sentence:
            i = random.randint(1, len(self.text))
            orig_sentence = self.lemmd_text[i]
        else:
            orig_sentence = self.lemmatize(input_sentence)
        trial_sent = orig_sentence.split()
        for word in trial_sent:
            if word not in self.model_keys:
                new_sent.append(word)
            else:
                new_sent.append(self.get_n_most_similar(word, n=1)[0][0])

        sent = ' '.join(new_sent)
        return orig_sentence, sent

