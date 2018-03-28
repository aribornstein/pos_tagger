"""
Written by Ari Bornstein
"""
import itertools
import collections
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data

global W_MAP, VOCAB, VECS 
W_MAP = {}
VOCAB = []
VECS = None

MINIBATCH = 100
EPOCHS = 15

START = "<s>"
END = "</s>"
UNK  = "UUUNKKK"
SPACE = " "
TAB = "\t"
NEWLINE = "\n"
CARRIGE_RETURN = "\n\n"

def read_input_data(fname):
    """
    Read input file and convert in to X and y data
    """
    f_data = open(fname).read().strip().replace(TAB, SPACE)
    X = []
    y = []
    sentences = f_data.split(CARRIGE_RETURN)
    for sen in sentences:
        words, tags = map(list, zip(*[token.split(SPACE) for token in sen.split(NEWLINE)]))
        X.append(words)
        y.append(tags)
    y = list(itertools.chain(*y))
    return (X, y)

def read_test_set(fname):
    """
    Read test set and convert in to X data
    """
    f_data = open(fname).read().strip().replace(TAB, SPACE)
    X = []
    sentences = f_data.split(CARRIGE_RETURN)
    for sen in sentences:
        words = sen.split(NEWLINE)
        X.append(words)
    return [X, list(itertools.chain(*X))]

def get_ixs(word, n=3):
    """
    Returns word and it's suffix and prefix of size n
    """
    return ["{}_pre".format(word[:n]), word, "{}_suf".format(word[-n:])]

def generate_vocab(sentences, pretrained=False, ixs=False, pruning=1):
    """
    Generate a vocab dictionary mapping words to id
    Prunes words that appear less than 5 times to train UNK
    """
    global W_MAP, VOCAB, VECS
    words = itertools.chain(*sentences)
    word_count = collections.Counter(words)
    VOCAB = list(set([w for w in word_count if word_count[w] > pruning]) | {START, END, UNK})
    W_MAP = {VOCAB[i]:i for i in range(len(VOCAB))}
    VECS = np.matrix(np.zeros(shape=(len(VOCAB), 50)))

    if pretrained:
        pretrained_words = open("vocab.txt").read().strip().split("\n")
        pretrained_vecs = np.loadtxt("wordVectors.txt")
        VOCAB = list(set(VOCAB) | set(pretrained_words))
        W_MAP = {VOCAB[i]:i for i in range(len(VOCAB))}
        VECS = np.matrix(np.zeros(shape=(len(VOCAB), 50)))
        for i, word in enumerate(pretrained_words):
            if word in W_MAP:
                VECS[W_MAP[word]] = pretrained_vecs[i]

    if ixs:
        pres = list(set(["{}_pre".format(word[:ixs]) for word in W_MAP]))
        sufs = list(set(["{}_suf".format(word[-ixs:]) for word in W_MAP]))
        ix_vocab = pres + sufs
        ix_vecs = np.zeros(shape=(len(ix_vocab), 50))
        for i, ix in enumerate(ix_vocab):
            W_MAP[ix] = len(W_MAP)
        VECS =  np.vstack([VECS, ix_vecs])
        VOCAB += ix_vocab

    VECS = VECS.flatten().reshape(len(W_MAP), 50)
def get_features(sentences, ixs=None):
    """
    Extracts training and DEVing features from a list of words
    """
    global W_MAP, VECS
    feature_list = []
    for sen in sentences:
        pp_w = p_w = START
        n = len(sen)
        for i in range(n):
            n_w = sen[i+1] if i+1 < n else END
            nn_w = sen[i+2] if i+2 < n else END
            word_features = [pp_w, p_w, sen[i], n_w, nn_w]
            if ixs:
                word_features = list(itertools.chain(*[get_ixs(wf) for wf in word_features]))
            word_feature_ids = [W_MAP[wf] if wf in W_MAP\
                                else W_MAP[UNK] for wf in word_features]
            feature_list.append(word_feature_ids)
            pp_w = p_w
            p_w = sen[i]

    return feature_list

def generate_input_data(train_file, dev_file, pretrained=False, ixs=None):
    """
    Generates tagging model data (vocab, tags, tags2id,
    trainloader, dev_x, dev_y)from input files.
    """
    train_x, train_y = read_input_data(train_file)
    dev_x, dev_y = read_input_data(dev_file)
    # Extract Vocab and Tag info
    generate_vocab(train_x, pretrained=pretrained, ixs=ixs)
    tags = list(set([t for t in train_y]))
    tags2id = {tags[i]:i for i in range(len(tags))}
    # Extract train and dev features
    train_x, train_y = get_features(train_x, ixs), [tags2id[t] for t in train_y]
    dev_x, dev_y = get_features(dev_x, ixs), [tags2id[t] for t in dev_y]
    train_x, train_y, dev_x, dev_y = map(torch.LongTensor, [train_x, train_y, dev_x, dev_y])

    # Extract loader
    trainset = data.TensorDataset(train_x, train_y)
    trainloader = data.DataLoader(trainset, batch_size=MINIBATCH, shuffle=True, num_workers=4)
    # Extract Test
    return [tags, trainloader, dev_x, dev_y]

def accuracy(y_pred, y, tags):
    """
    Returns the accuracy of a classifier 
    """
    o_id = tags.index("O") if "O" in tags else None
    correct = ignore = 0
    for i, tag_id in enumerate(y):
        if y_pred[i] == y[i]:
            if tag_id == o_id:
                ignore += 1
            else:
                correct += 1
    return float(correct)/(len(y) - ignore)

def get_metrics_data(model, x, y):
    """
    returns labels and predicted values
    in format for sci-kit learn metrics
    """
    outputs = model(Variable(x).cuda())
    _, predicted = torch.max(outputs.data, 1)
    labels = y.cpu().numpy()
    predicted = predicted.cpu().numpy()
    return labels, predicted

def write_model_results(model, test_file, tags, outpath, ixs = False):
    """
    Output model results on the test set
    """
    test, test_data = read_test_set(test_file)
    test_x = get_features(test, ixs)
    predicted = model(Variable(torch.LongTensor(test_x).cuda()))
    _, labels = torch.max(predicted.data, 1)
    predictions = NEWLINE.join(["{} {}".format(test_data[i], tags[labels[i]])\
                                 for i in range(len(test_data))])
    with open(outpath, "w") as outfile:
        outfile.write(predictions)

if __name__ == "__main__":
    pass
