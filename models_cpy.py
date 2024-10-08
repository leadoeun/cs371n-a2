# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


    

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
    def __init__(self, inp, hid, out, embedding):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(NNModule, self).__init__()
        self.V = nn.Linear(inp, hid)
        
        # self.g = nn.Tanh()
        # self.g = nn.ReLU()
        self.g = nn.Sigmoid()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)



    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        # print(self.V(x))
        # print(self.W(self.g(self.V(x))).shape)

        return self.log_softmax(self.W(self.g(self.V(x)))) 
    
    def init_embedding(self, embedding):
        return embedding.get_initialized_embedding_layer()
    def get_loss(self):
        return nn.NLLLoss()



    

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
    def __init__(self, nn_module, word_embeddings, embedding):
        self.nn_module = nn_module
        self.word_embeddings = word_embeddings
        self.embedding = embedding

    def predict(self, ex_words:List[str]) -> int:
        x_temp = []
        for word in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(word)
            if idx == -1 :
                idx = self.word_embeddings.word_indexer.index_of("UNK")
            x_temp.append(idx)
        # x_temp = [x_temp]
        x_temp = torch.LongTensor(x_temp)
        x_temp = self.embedding(x_temp)
        x_temp = x_temp.mean(dim=0)
        
        log_probs = self.nn_module.forward(x_temp)
        return torch.argmax(log_probs)



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # indexer = Indexer()
    # for i, sentence in enumerate(train_exs):
    #     for word in sentence.words:
    #         if indexer.contains(word):
    #             continue
    #         else:
    #             indexer.add_and_get_index(word)
    

    # inp = len(indexer)


    inp = 0
    if args.word_vecs_path == 'data/glove.6B.50d-relativized.txt':
        inp = 50
    else:
        inp = 300

    

    epochs = 100
    learning_rate = 0.0001
    embedding_size = 100
    num_class = 2
    batch_size = args.batch_size
    
    

    ffnn = NNModule(inp, embedding_size, num_class, word_embeddings)
    embedding = ffnn.init_embedding(word_embeddings)
    cal_loss = ffnn.get_loss()


            
    
    

    optimizer = optim.Adam(ffnn.parameters(), lr=learning_rate)
    # random.seed(1)
    for epoch in range(0, epochs):
        x = []
        y = []
        random.shuffle(train_exs)
        
        # for i, sentence in enumerate(train_exs):
        #     x_temp = []
        #     for word in sentence.words:
        #         idx = word_embeddings.word_indexer.index_of(word)
        #         if idx == -1 :
        #             idx = word_embeddings.word_indexer.index_of("UNK")
        #         x_temp.append(idx)
        #     x_temp = torch.LongTensor(x_temp)
        #     x_temp = embedding(x_temp)
        #     x.append(x_temp.mean(dim=0))
        #     y.append(sentence.label)

        for i in range(0, len(train_exs), batch_size):
            x_temp = []
            y_temp = []
            max = 0
            for j in range(0, batch_size):
                if len(train_exs[i+j].words) > max:
                    max = len(train_exs[i+j].words)
            for j in range(0, batch_size):
                batch_temp = []
                sentence = train_exs[i+j]
                for word in sentence.words:
                    idx = word_embeddings.word_indexer.index_of(word)
                    if idx == -1 :
                        idx = word_embeddings.word_indexer.index_of("UNK")
                    batch_temp.append(idx)
                while len(batch_temp) < max:
                    batch_temp.append(word_embeddings.word_indexer.index_of("PAD"))
                
                # batch_temp = torch.LongTensor(batch_temp)
                # batch_temp = embedding(batch_temp)
                # print(batch_temp)
                # x_temp.append(batch_temp.mean(dim=0).tolist())
                x_temp.append(batch_temp)
                
                y_temp.append(sentence.label)
            # if i==0: print(x_temp)
            x_temp = torch.LongTensor(x_temp)
            x_temp = embedding(x_temp)
            # if i==0: print(x_temp.shape)
            x_temp = torch.mean(x_temp, dim=1)

            # if i==0: print(x_temp)
            # print(x_temp.shape)
            # print(x_temp)
            # if i==0: print(x_temp)
            if batch_size==1:
                x.append(x_temp[0])
            else:
                x.append(x_temp)
            y.append(y_temp)
                # batch_temp = torch.LongTensor(batch_temp)
                # batch_temp = embedding(batch_temp)
                # x_temp.append(batch_temp.mean(dim=0))
            #     x_temp.append(batch_temp)
            #     y_temp.append(sentence.label)
            # x_temp = torch.LongTensor(x_temp)

            # x_temp = embedding(x_temp)
            
            # x.append(x_temp.mean(dim=0))
            # if i==0: print(x[0].shape)
            # y.append(y_temp)

        

        
        
        tot_loss = 0.0
        for i in range(0, len(x)):
            
            cur_x = x[i]
            # if i==0: print(cur_x)
            # print(cur_x)
            cur_y = y[i]
            # print(cur_y)

            y_onehot = torch.zeros((batch_size, num_class))

            for j in range(0, batch_size):
                y_onehot[j].scatter_(0, torch.from_numpy(np.asarray(cur_y[j],dtype=np.int64)), 1)

            # y_onehot.scatter_(0, torch.from_numpy(np.asarray(cur_y,dtype=np.int64)), 1)
            # print(y_onehot)
            
            
            ffnn.zero_grad()
            log_probs = ffnn.forward(cur_x)
            # loss = 0

            # loss = torch.neg(log_probs).dot(y_onehot)
            # for j in range(0, batch_size):
            #     print(log_probs)
            #     print(y_onehot)
            #     loss += cal_loss(log_probs[j], y_onehot[j])
            loss = torch.mean(torch.matmul(torch.neg(log_probs), torch.transpose(y_onehot, 0, 1)))
            
            # if i==0:
            #     print(log_probs)
            #     print(y_onehot)
            #     print(loss)
            
            
            tot_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, tot_loss))
    return NeuralSentimentClassifier(ffnn, word_embeddings, embedding)
    




    # raise NotImplementedError

