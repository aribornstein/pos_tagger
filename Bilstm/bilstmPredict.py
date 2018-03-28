"""
Written by Ari Bornstein
"""
import argparse
import utils
import torch
import itertools
from bilstmTrain import *


def write_model_results(model, input_file, repr, tags, outpath):
    """
    Output model results on the test set
    """
    input, input_data = read_input(input_file)

    if repr == "c":
        x = utils.get_features(input, ixs=3)
    else:
        x = utils.get_features(input, chars=True)

    w_batcher = utils.AutoBatcher(x, x, batch_size=1, shuffle=False)
    labels = []
    for inputs, _ in w_batcher.get_batches():
        output = torch.max(model(inputs), 1)[1]
        labels += output.cpu().data.numpy().tolist()

    predictions = utils.NEWLINE.join(["{} {}".format(input_data[i], tags[labels[i]])\
                                 for i in range(len(input_data))])
    with open(outpath, "w") as outfile:
        outfile.write(predictions)

def read_input(fname):
    """
    Read test set and convert in to X data
    """
    f_data = open(fname).read().strip().replace(utils.TAB, utils.SPACE)
    X = []
    sentences = f_data.split(utils.CARRIGE_RETURN)
    for sen in sentences:
        words = sen.split(utils.NEWLINE)
        X.append(words)
    return [X, list(itertools.chain(*X))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tag an input file text file a pretrained bilstm for pos and ner.')
    parser.add_argument("repr")
    parser.add_argument("modelFile")
    parser.add_argument("inputFile")

    args = parser.parse_args()

    MODEL = torch.load(args.modelFile)

    utils.W_MAP, utils.C_MAP = MODEL.maps
    POS_TAGS = ['PRP$', 'VBG', 'VBD', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', '#', 'RP', '$', 'NN',
                ')', '(', 'FW', 'POS', '.', 'TO', 'PRP', 'RB', ':', 'NNS', 'NNP', '``', 'WRB', 'CC', 'LS', 'PDT', 'RBS',
                'RBR', 'CD', 'EX', 'IN', 'WP$', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'VB', 'UH']
    NER_TAGS = ['ORG', 'MISC', 'PER', 'O', 'LOC']

    OUT = "{}_{}.out".format(args.inputFile, args.repr)
    if len(POS_TAGS) == MODEL.out_dim:
        write_model_results(MODEL, args.inputFile, args.repr, POS_TAGS, OUT)
    else:
        write_model_results(MODEL, args.inputFile, args.repr, NER_TAGS, OUT)


