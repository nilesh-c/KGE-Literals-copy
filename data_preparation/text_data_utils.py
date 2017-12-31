import re
import sys
import itertools
import numpy as np
from collections import Counter
import pdb

"""
Adapted from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_text(text_data):
    text_data = [s.strip() for s in text_data]
    text_data = [clean_str(sent) for sent in text_data]
    return text_data

def pad_sentences(sentences, max_len=None, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if max_len==None:
        max_len = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if max_len>len(sentence):
            num_padding = max_len - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_len]

        padded_sentences.append(new_sentence)
    return padded_sentences

def prepare_vocab(text_data, max_len = None):

    x_text = preprocess_text(text_data)
    x_text = [s.split(" ") for s in x_text]
    text_padded = pad_sentences(x_text, max_len)
    # Build vocabulary
    word_counts = Counter(itertools.chain(*text_padded))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}        
    return vocabulary

def build_data(text_data, vocabulary, max_len=None):

    text_data = preprocess_text(text_data)
    text_data = [s.split(" ") for s in text_data]
    text_padded = pad_sentences(text_data, max_len)
    text_data = np.array([[vocabulary[word] for word in text] for text in text_padded])

    return text_data