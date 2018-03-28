"""
Written by Ari Bornstein
"""

import torch
from torch import nn
import itertools
from torch.autograd import Variable
from torch.nn import functional as F
import utils
import time
import argparse

class ReprA(nn.Module):
    """
    A Feed forward network for tagging
    """

    def __init__(self, vocab_size, out_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ReprA, self).__init__()
        self.out_dim = out_dim
        self.embeddings = torch.nn.Embedding(vocab_size, out_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(utils.VECS))

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        inputs = [list(itertools.chain(*w)) for w in [sen[0:][::2] for sen in x]]
        inputs = Variable(torch.LongTensor(inputs)).cuda()
        return self.embeddings(inputs)

class ReprB(nn.Module):
    """
    A Feed forward network for tagging
    """
    def __init__(self, charset_size, embed_dim, out_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ReprB, self).__init__()
        self.out_dim = out_dim
        self.char_embeddings = torch.nn.Embedding(charset_size, embed_dim)
        self.char_rnn = nn.LSTM(embed_dim, out_dim, batch_first=True, num_layers=2)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = [list(itertools.chain(*w)) for w in [sen[1:][::2] for sen in x]]

        sen_length = len(x[0])
        for sen in x:
            assert len(sen) == sen_length  # Sentences must be uniform in size

        w_chars = [word_seq for sen in x for word_seq in sen]  # concat lists
        org_ids = torch.arange(0, len(w_chars)).long()
        w_batcher = utils.AutoBatcher(w_chars, org_ids, shuffle=True, batch_size=50)
        w_features = []
        w_ids = []
        for chars, char_ids in w_batcher.get_batches():
            char_var = Variable(torch.LongTensor(chars)).cuda()
            char_embeddings = self.char_embeddings(char_var)
            lstm_out, __ = self.char_rnn(char_embeddings)
            char_features = lstm_out[:, -1, :]
            w_features.extend(char_features)
            w_ids.extend(char_ids)

        # rearrange
        reverse_ids = [i for i, ids in sorted([(i, ids) for i, ids in enumerate(w_ids)], key=lambda (i, ids): ids)]
        stacked_wfs = torch.stack(w_features)[reverse_ids, :]
        sen_wfs = torch.split(stacked_wfs, sen_length, 0)
        return torch.stack(sen_wfs)

class ReprC(ReprA):
    """
    A Feed forward network for tagging
    """

    def __init__(self, vocab_size, out_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ReprC, self).__init__(vocab_size, out_dim)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        inputs = Variable(torch.LongTensor(x)).cuda()
        embedding = self.embeddings(inputs)
        seq_size, ixs_size = [inputs.size()[1] / 3, 3]
        ixs = embedding.view(inputs.size()[0], seq_size, ixs_size, -1)  # break embed rows into ix tensor for each word
        sum_ixs = ixs.view(inputs.size()[0], seq_size, ixs_size, -1).sum(2)  # sum ixs together to get rows of word embed vectors
        return sum_ixs

class ReprD(nn.Module):
    """
    A Feed forward network for tagging
    """
    def __init__(self, charset_size, vocab_size, out_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ReprD, self).__init__()
        self.out_dim = out_dim*2
        self.reprA = ReprA(vocab_size, out_dim)
        self.reprB = ReprB(charset_size, 20, out_dim)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        a = self.reprA(x)
        b = self.reprB(x)
        return torch.cat([a, b], 2)

class BiLSTM_Tagger(nn.Module):
    """
    A Feed forward network for tagging
    """
    def __init__(self, reprW, hidden_dim, out_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(BiLSTM_Tagger, self).__init__()
        self.out_dim = out_dim
        self.reprW = reprW
        self.rnn = nn.LSTM(reprW.out_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.maps = [utils.W_MAP, utils.C_MAP]

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        repr = self.reprW(x)
        out, __ = self.rnn(repr)
        out = F.tanh(self.linear1(out))
        out = self.linear2(out)
        return out.view(-1, self.out_dim)

def train_model(train_file, dev_file, outpath, repr, epochs):
    """
    trains a tagging model
    """
    train_stats = []
    if repr == "c":
        tags, train_batcher, X_dev, y_dev = utils.generate_input_data(train_file, dev_file, ixs=3, pretrained=True)
    else:
        tags, train_batcher, X_dev, y_dev = utils.generate_input_data(train_file, dev_file, pretrained=True, chars=True)
    V = len(utils.VOCAB)
    C = len(utils.CHARS)
    E = 25 # Char Embedding dimensions
    R = 50 # Representation dimensions
    H = 128 # hidden layers
    D_out = len(tags) # out layer

    # Choose our representation
    if repr == "a":
        reprW = ReprA(V, R) # part 1
    if repr == "b":
        reprW = ReprB(V, E, R)  # part 2
    if repr == "c":
        reprW = ReprC(V, R) # part 3
    if repr == "d":
        reprW = ReprD(C, V, R) # part 4

    #Init our model
    model = BiLSTM_Tagger(reprW, H, D_out)
    criterion = torch.nn.CrossEntropyLoss() # cross entropy loss
    optimizer = torch.optim.Adam(model.parameters()) # ADAM
    start = time.time()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_time = time.time()
        train_size = train_batcher.batch_count()
        data_count = 0
        for i, data in enumerate(train_batcher.get_batches(), 0):
            data_count += len(data)
            if data_count > 500:
                labels, predicted = utils.get_metrics_data(model, X_dev, y_dev)
                train_stats.append({"Epoch": epoch + 1, "Loss": running_loss / train_size,
                                    "Dev": utils.accuracy(labels, predicted, tags)})
                data_count -= 500

            inputs, labels = data
            if torch.cuda.is_available():
                model.cuda()
                labels = Variable(torch.LongTensor(labels)).cuda()
            else:
                labels = Variable(torch.LongTensor(labels))

            labels = torch.cat(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        # epoch stats
        end = time.time()
        labels, predicted = utils.get_metrics_data(model, X_dev, y_dev)
        print('Epoch [%d] loss: %.3f dev: %.3f epoch time %f runtime %f' % (epoch + 1, running_loss / train_size,
                                                                            utils.accuracy(labels, predicted, tags),
                                                                            end - epoch_time, end - start))
    #Save model
    torch.save(model, outpath)
    return train_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a bilstm for pos and ner.')
    parser.add_argument("repr")
    parser.add_argument("train")
    parser.add_argument("model")
    parser.add_argument("dev")

    args = parser.parse_args()

    stats = train_model(args.train, args.dev, args.model, args.repr, utils.EPOCHS)
