# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class NewWordEmbeddings(WordEmbeddings):
    def indexer_init(self, train_exs):
        for sentence in train_exs:
            self.vectorize(sentence.words)

    
    def vectorize(self, sentence: List[str]):
        for word in sentence:
            idx = self.word_indexer.index_of(word)
            self.vectors[idx] += 1
        return self.vectors




    

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]
    
class NNModule(nn.Module):
    def __init__(self, inp, hid, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(NNModule, self).__init__()
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)
        # nn.Embedding(inp,hid)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        return self.log_softmax(self.W(self.g(self.V(x))))  
    
    # def embed(self, inp, hid):
    #     return 

def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, nn_module, word_embeddings):
        self.nn_module = nn_module
        self.word_embeddings = word_embeddings

    def predict(self, ex_words:List[str]) -> int:
        indexer = self.word_embeddings.word_indexer
        for word in ex_words:
            if indexer.contains(word):
                print(word)
                continue
            idx = indexer.index_of(word)
            vector = np.zeros(len(indexer))
            vector[idx] += 1
            self.word_embeddings.vectors = vector
        x = self.word_embeddings.vectors
        x = form_input(x)
        log_probs = self.nn_module.forward(x)
        return torch.argmax(log_probs)



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    indexer = Indexer()
    for i, sentence in enumerate(train_exs):
        for word in sentence.words:
            if indexer.contains(word):
                continue
            else:
                indexer.add_and_get_index(word)
    

    inp = len(indexer)
    x = np.full((len(train_exs), inp), fill_value = 0, dtype = float)
    y = np.full(len(train_exs), fill_value=-1, dtype = float)
    print(inp)
    for i, sentence in enumerate(train_exs):
        for word in sentence.words:
            idx = indexer.index_of(word)
            vector = np.zeros(inp)
            vector[idx] += 1
            word_embeddings.vectors = vector
        x[i] = vector
        y[i] = sentence.label
    epochs = 100
    learning_rate = 0.1
    embedding_size = 4
    num_class = 2
    
    ffnn = NNModule(inp, embedding_size, num_class)
    optimizer = optim.Adam(ffnn.parameters(), lr=learning_rate)
    for epoch in range(0, epochs):
        random.shuffle(train_exs)
        total_loss = 0.0
        for i in range(0, len(train_exs)):
            cur_x = form_input(x[i])
            cur_y = y[i]
            y_onehot = torch.zeros(num_class)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(cur_y,dtype=np.int64)), 1)
            ffnn.zero_grad()
            log_probs = ffnn.forward(cur_x)
            loss = torch.neg(log_probs).dot(y_onehot)

            total_loss += loss
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
    return NeuralSentimentClassifier(ffnn, word_embeddings)
    




    # raise NotImplementedError

