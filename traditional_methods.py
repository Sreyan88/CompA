import torch
import random
import numpy as np
from functools import partial
import torch.nn.functional as nnf
from torchvision import transforms as T

class TextShuffler:

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def shuffle_nouns_and_adj(self, ex):

        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN']]
        ## Finding adjectives
        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        ## Shuffle the nouns of the text
        text[noun_idx] = np.random.permutation(text[noun_idx])
        ## Shuffle the adjectives of the text
        text[adjective_idx] = np.random.permutation(text[adjective_idx])

        return " ".join(text)

    def count_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['VBG']]
        ## Finding adjectives

        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        #get source
        print(text[noun_idx])
        print(noun_idx)
        return noun_idx

    def count_prep(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['ADP']]
        ## Finding adjectives

        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        #get source
        print(text[noun_idx])
        print(noun_idx)
        return noun_idx

    def count_nouns(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN']]
        ## Finding adjectives

        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        #get source
        #print(text[noun_idx])
        #print(noun_idx)
        return noun_idx

    def shuffle_all_words(self, ex):
        return " ".join(np.random.permutation(ex.split(" ")))


    def shuffle_allbut_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
        ## Finding adjectives

        else_idx = np.ones(text.shape[0])
        else_idx[noun_adj_idx] = 0

        else_idx = else_idx.astype(bool)
        ## Shuffle everything that are nouns or adjectives
        text[else_idx] = np.random.permutation(text[else_idx])
        return " ".join(text)


    def get_trigrams(self, sentence):
        # Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
        trigrams = []
        trigram = []
        for i in range(len(sentence)):
            trigram.append(sentence[i])
            if i % 3 == 2:
                trigrams.append(trigram[:])
                trigram = []
        if trigram:
            trigrams.append(trigram)
        return trigrams

    def trigram_shuffle(self, sentence):
        trigrams = self.get_trigrams(sentence)
        for trigram in trigrams:
            random.shuffle(trigram)
        return " ".join([" ".join(trigram) for trigram in trigrams])


    def shuffle_within_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        shuffled_ex = self.trigram_shuffle(tokens)
        return shuffled_ex


    def shuffle_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        trigrams = self.get_trigrams(tokens)
        random.shuffle(trigrams)
        shuffled_ex = " ".join([" ".join(trigram) for trigram in trigrams])
        return shuffled_ex