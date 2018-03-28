import sys
import collections
import numpy as np
from config import SPACE, NEW_LINE, START_SIGHT, BUFFER_SIZE, TAB, SLASH, UNK, END

WORD_INDEX = 0
TAG_INDEX = 1

def write(outpath, output):
    text_file = open(outpath, "w")
    text_file.write(str(output))
    text_file.close()

class MLE_Trainer:
    """
    A Class that calculates e and q for HMM models
    """
    def __init__(self, corpus=None, q_mle_out = None, e_mle_out = None):
        self.tokens = None
        self.tags = None
        self.bigram_tags = None
        self.trigram_tags = None

        if corpus:
            self.train(corpus)
        if e_mle_out:
            self.write_e(e_mle_out)
        if q_mle_out:
            self.write_q(q_mle_out)
    
    def train(self, corpus):
        """
        Processes a raw corpus to generate the statistics needed for 
        genereating e and q 
        """
        self.tokens = []
        self.tags = []
        sentences = corpus.split(NEW_LINE)
        for sentence in sentences:
            start = START_SIGHT + SLASH + START_SIGHT + SPACE + START_SIGHT + SLASH + START_SIGHT + SPACE
            end = SPACE + END + SLASH + END
            sentence = start  + sentence + end 
            tokens = sentence.split(SPACE)
            for t in tokens:
                token = t.rsplit(SLASH, 1)
                if (len(token) > 1):
                    self.tokens.append(token) 
                    self.tags.append(token[TAG_INDEX])
        
        nonsense_cases = set([(END, START_SIGHT), (START_SIGHT, END),
                              (START_SIGHT, START_SIGHT, END),
                              (END, START_SIGHT, START_SIGHT)])
        self.bigram_tags = [b for b in zip(self.tags[:-1], self.tags[1:]) if b not in nonsense_cases]
        self.trigram_tags = [t for t in zip(self.tags[:-1], self.tags[1:], self.tags[2:])\
                            if not (t[WORD_INDEX], t[TAG_INDEX]) in nonsense_cases and\
                            not (t[WORD_INDEX], t[TAG_INDEX]) in nonsense_cases]

    def write_e(self, outpath):
        """
        Returns the number of tag instances for each word the vocabulary
        Optimize this by removing closed tags 
        """

        if not self.tokens:
            raise Exception("MLE model not yet trained")

        word_counts = collections.Counter([word_tag[WORD_INDEX] for word_tag in self.tokens])
        # Count and format word tag pairs with out unks
        e_tokens = [(token[WORD_INDEX], token[TAG_INDEX]) for token in self.tokens]
        e_counts = dict(collections.Counter(e_tokens))
        formatted_counts = [k[WORD_INDEX] + SPACE + k[TAG_INDEX] + TAB + str(e_counts[k]) for k in e_counts]
        output = NEW_LINE.join(formatted_counts)
        write(outpath, output)

    def write_q(self, outpath):
        """
        Returns the number of subsquent tag instances for each tag the vocabulary
        Optimize this by removing closed tags 
        """
        if not self.bigram_tags:
            raise Exception("MLE model not yet trained")

        # Count and format bigrams
        bigram_counts = dict(collections.Counter(self.bigram_tags))
        formatted_bigram_counts = [k[0] + SPACE + k[1] + TAB + str(bigram_counts[k]) for k in bigram_counts]
        bigram_output =  NEW_LINE.join(formatted_bigram_counts)
        # Count and format trigrams
        trigram_counts = dict(collections.Counter(self.trigram_tags))
        formatted_trigram_counts = [k[0] + SPACE + k[1] + SPACE + k[2] + TAB + str(trigram_counts[k]) for k in trigram_counts]
        trigram_output =  NEW_LINE.join(formatted_trigram_counts)
        # Write output
        output = bigram_output + NEW_LINE + trigram_output
        write(outpath, output)

        
if __name__ == "__main__":
    if len(sys.argv) > 3:
    
        # Train
        input_filename = sys.argv[1]
        q_mle_filename = sys.argv[2]
        e_mle_filename = sys.argv[3]

        TRAIN_DATA = open(input_filename, "r").read().encode('utf-8')
        MLE_TRAINER = MLE_Trainer(TRAIN_DATA, q_mle_filename,
                                  e_mle_filename)

    else:
        print "Inputs Invalid"