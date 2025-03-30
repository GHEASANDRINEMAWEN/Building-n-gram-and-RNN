#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   
# Copyright (C) 2024
# 
# @author: Ezra Fu <erzhengf@andrew.cmu.edu>
# based on work by 
# Ishita <igoyal@andrew.cmu.edu> 
# Suyash <schavan@andrew.cmu.edu>
# Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 2
N-gram Language Model Implementation

Complete the LanguageModel class and other TO-DO methods.
"""

#######################################
# Import Statements
#######################################
from utils import *
from collections import Counter
from itertools import product
import argparse
import random
import math

#######################################
# TODO: get_ngrams()
#######################################
def get_ngrams(list_of_words, n):
    """
    Returns a list of n-grams for a list of words.
    
    Args:
    ----
    list_of_words: List[str]
        List of already preprocessed and flattened (1D) list of tokens e.g. ["<s>", "hello", "</s>", "<s>", "bye", "</s>"]
    n: int
        n-gram order e.g. 1, 2, 3
    
    Returns:
    -------
    n_grams: List[Tuple]
        Returns a list containing n-gram tuples
    """
    n_grams = []
    for i in range(len(list_of_words) - n + 1):
        n_gram = tuple(list_of_words[i:i + n])
        n_grams.append(n_gram)
    return n_grams


#######################################
# TODO: NGramLanguageModel()
#######################################
class NGramLanguageModel():
    def __init__(self, n, train_data, alpha=1):
        """
        Language model class.

        Args:
        -----
        n: int
            n-gram order (e.g., 1, 2, 3).
        train_data: List[List]
            Preprocessed training data in sentence format.
        alpha: float
            Smoothing parameter.

        Attributes:
        -----
        self.tokens: Flattened list of tokens.
        self.vocab: Dictionary of word counts.
        self.model: Stores n-gram probabilities.
        self.n_grams_counts: Dictionary of n-gram counts.
        self.prefix_counts: Dictionary of (n-1)-gram counts.
        """
        self.n = n
        self.smoothing = alpha
        self.train_data = train_data

        # Preprocess data
        self.tokens = flatten(train_data)
        self.vocab = Counter(self.tokens)
        self.n_grams_counts = Counter()
        self.prefix_counts = Counter()
        self.model = {}

        # Construct model
        self.build()

    def build(self):
        """
        Computes n-gram and (n-1)-gram counts and stores initial probabilities.
        """
        ngram_list = get_ngrams(self.tokens, self.n)

        for ngram in ngram_list:
            self.n_grams_counts[ngram] += 1
            if self.n > 1:
                prefix = ngram[:-1]
                self.prefix_counts[prefix] += 1

        # Store precomputed probabilities
        for ngram in self.n_grams_counts.keys():
            self.model[ngram] = self.get_prob(ngram)

    def get_smooth_probabilities(self, ngrams):
        """
        Returns smoothed probabilities for a list of n-grams.

        Args:
        -----
        ngrams: list of tuples
            List of n-gram tuples.

        Returns:
        -----
        list of float
            List of probabilities.
        """
        return list(map(self.get_prob, ngrams))

    def get_prob(self, ngram):
        """
        Computes probability of an n-gram using Laplace smoothing.

        Args:
        -----
        ngram: tuple
            The n-gram tuple.

        Returns:
        -----
        float
            Probability of the n-gram.
        """
        vocab_size = len(self.vocab)
        count = self.n_grams_counts.get(ngram, 0)

        if self.n == 1:
            return (count + self.smoothing) / (sum(self.vocab.values()) + self.smoothing * vocab_size)

        prefix = ngram[:-1]
        prefix_count = self.prefix_counts.get(prefix, 0)
        return (count + self.smoothing) / (prefix_count + self.smoothing * vocab_size)

    def perplexity(self, test_data):
        """
        Computes perplexity using lazy Laplace smoothing.

        Args:
        -----
        test_data: List[List]
            Nested list of preprocessed test sentences.

        Returns:
        -----
        float
            Perplexity score.
        """
        test_tokens = flatten(test_data)
        test_ngrams = get_ngrams(test_tokens, self.n)

        if not test_ngrams:
            return float("inf")

        log_prob_total = sum(math.log(self.model.get(ngram, self.get_prob(ngram)) + 1e-10) for ngram in test_ngrams)
        return math.exp(-log_prob_total / len(test_ngrams))


###############################################
# Method: Most Probable Candidates [Don't Edit]
###############################################

# Copy of main executable script provided locally for your convenience
# This is not executed on autograder, so do what you want with it
if __name__ == '__main__':
    train = "data/sample.txt"
    test = "data/sample.txt"
    n = 2
    alpha = 0

    print("No of sentences in train file: {}".format(len(train)))
    print("No of sentences in test file: {}".format(len(test)))

    print("Raw train example: {}".format(train[2]))
    print("Raw test example: {}".format(test[2]))

    train = preprocess(train, n)
    test = preprocess(test, n)

    print("Preprocessed train example: \n{}\n".format(train[2]))
    print("Preprocessed test example: \n{}".format(test[2]))

    # Language Model
    print("Loading {}-gram model.".format(n))
    lm = NGramLanguageModel(n, train, alpha)

    print("Vocabulary size (unique unigrams): {}".format(len(lm.vocab)))
    print("Total number of unique n-grams: {}".format(len(lm.model)))
    
    # Perplexity
    ppl = lm.perplexity(test_data=test)
    print("Model perplexity: {:.3f}".format(ppl))
    
    # Generating sentences using your model
    print("Generating random sentences.")
    num_to_generate = 5
    for sentence, prob in generate_sentences(lm, num_to_generate):
        print("{} ({:.5f})".format(sentence, prob))