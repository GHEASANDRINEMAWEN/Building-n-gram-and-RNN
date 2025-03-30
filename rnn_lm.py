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
RNN Language Model Implementation

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


import torch
import torch.nn as nn
import torch.nn.functional as F  # Import F for softmax and other functions

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        """
        RNN model class.
        
        Args
        ____
        vocab_size: int
            Size of the vocabulary
        embedding_dim: int
            Dimension of the word embeddings
        hidden_dim: int
            Dimension of the hidden state of the RNN
        embedding_matrix: torch.Tensor
            Pre-trained GloVe embeddings
        """
        super(RNNLanguageModel, self).__init__()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                 else "cuda" if torch.cuda.is_available() 
                                 else "cpu")
        print(f"Using device: {self.device}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer initialized with GloVe embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Freeze the embeddings

        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.to(self.device)

    def forward(self, x, hidden=None):
        """
        The forward pass of the RNN model.
        
        Args
        ____
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length)
        hidden: torch.Tensor
            Hidden state tensor of shape (num_layers, batch_size, hidden_dim)
        
        Returns
        -------
        out: torch.Tensor
            Output tensor of shape (batch_size, sequence_length, vocab_size)
        hidden: torch.Tensor
            Hidden state tensor of shape (num_layers, batch_size, hidden_dim)
        """
        x = x.to(self.device)
        if hidden is None:
            # Initialize hidden state if not provided
            hidden = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
        else:
            hidden = hidden.to(self.device)

        # Embedding lookup
        x = self.embedding(x)

        # RNN forward pass
        out, hidden = self.rnn(x, hidden)

        # Fully connected layer
        out = self.fc(out)

        return out, hidden

    def generate_sentence(self, sequence, word_to_ix, ix_to_word, num_words, mode='max'):
        """
        Predicts the next words given a sequence.
        
        Args
        ____
        sequence: str
            Input sequence
        word_to_ix: dict
            Dictionary mapping words to their corresponding indices
        ix_to_word: dict
            Dictionary mapping indices to their corresponding words
        num_words: int
            Maximum number of words to predict
        mode: str
            Mode of prediction. 'max' or 'multinomial'
            'max' mode selects the word with maximum probability
            'multinomial' mode samples the word from the probability distribution
        
        Returns
        -------
        predicted_sequence: List[str]
            List of predicted words (excluding the initial sequence)
        """
        self.eval()
        predicted_sequence = []

        # Convert the input sequence to indices, handling unknown words
        sequence_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in sequence.split()]
        sequence_tensor = torch.tensor(sequence_indices, dtype=torch.long).unsqueeze(0).to(self.device)

        hidden = None
        for _ in range(num_words):
            with torch.no_grad():
                # Forward pass
                output, hidden = self.forward(sequence_tensor, hidden)
                output = output[:, -1, :]  # Get the last output

                # Predict the next word
                if mode == 'max':
                    _, next_word_idx = torch.max(output, dim=1)
                elif mode == 'multinomial':
                    probs = F.softmax(output, dim=1)  # Use F.softmax
                    next_word_idx = torch.multinomial(probs, num_samples=1).squeeze()

                # Get the predicted word
                next_word = ix_to_word.get(next_word_idx.item(), '<UNK>')
                predicted_sequence.append(next_word)

                # Update the sequence tensor with the predicted word
                sequence_tensor = torch.tensor([[next_word_idx.item()]], dtype=torch.long).to(self.device)

        return predicted_sequence