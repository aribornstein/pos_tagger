import sys
import collections
import numpy as np
from config import SPACE, NEW_LINE, START_SIGHT, BUFFER_SIZE, TAB, SLASH, UNK, END, HMM_PATTERNS
# from ../accuracy_checker import check_accuracy  

class GreedyTagger:
    """
    A Class that calculates e and q for HMM models
    """
    def __init__(self, q_mle, e_mle):
        self.tags = set()
        self.q_dict = {}
        self.e_dict = {}
        self.e_cache = {}
        self.q_cache = {}
        self.known_words_set = set()
        self.tag_count_dict = {}
        self.total_word_count = None
            
        # read the q and e tables to local dicts
        self.generate_q_dict(q_mle)
        self.generate_e_dict(e_mle)

        # initialize the dictionary of tags (contains it's count)
        for word_tag_tuple in self.e_dict.keys():
            tag = word_tag_tuple[1]
            if tag in self.tag_count_dict:
                self.tag_count_dict[tag] += self.e_dict[word_tag_tuple]
            else:
                self.tag_count_dict[tag] = self.e_dict[word_tag_tuple]

        self.total_word_count = sum([self.tag_count_dict[key] for key in self.tag_count_dict.keys()])
        print "Total Word Count is: {0}".format(self.total_word_count)

        # initialize the tags set and known words set
        self.tags.update(self.tag_count_dict.keys())
        self.known_words_set.update([word_tag_tuple[0] for word_tag_tuple in self.e_dict.keys()])


    def most_likley_tag(self, word):
        """
        Returns the most likely tag for unknown word
        """
        for p in HMM_PATTERNS:
            if p[0].match(word):
                return p[1]
        return 'NNP'

    def handle_unk(self, word, tag):
        """
        Handles unknown words
        """        
        if word in self.known_words_set:
            return 1.0 / self.total_word_count
        else:
             return 1.0 if (self.most_likley_tag(word) == tag) else (1.0 / self.total_word_count)
    
    def calculate_e(self, word, tag):
        """
        takes in a list of word counts and returns the emmision probailites
        add unknown handling and common signatures
        """
        if (word, tag) in self.e_dict:
            return float(self.e_dict[(word,tag)]) / self.tag_count_dict[tag]
        else:
            return self.handle_unk(word, tag)

    def calculate_q(self, prev_prev, prev, tag):
        """
        takes in a list of word counts and returns the transitions probailites
        add unknown handling 
        """
        if tag != START_SIGHT and tag != END:
            if (prev_prev, prev, tag) in self.q_dict:
                pr = 0.8 * float(self.q_dict[(prev_prev, prev, tag)]) / self.q_dict[(prev_prev, prev)]
                pr += 0.15 * float(self.q_dict[(prev, tag)]) / self.tag_count_dict[prev]
                pr += 0.05 * float(self.tag_count_dict[tag]) / self.total_word_count
                return pr

            elif (prev, tag) in self.q_dict:
                pr = 0.8 * float(self.q_dict[(prev, tag)]) / self.tag_count_dict[prev]
                pr += 0.2 * float(self.tag_count_dict[tag]) / self.total_word_count
                return pr

        return float(self.tag_count_dict[tag]) / self.total_word_count

    def parse_line(self, line):
        """
        Parses a line of tokens
        """
        data, count = line.rsplit(TAB, 1)
        data = data.split(SPACE)
        return tuple(data), float(count)

    def generate_q_dict(self, q_mle):
        """
        Process the tag-gram counts from q_mle
        """
        with open(q_mle, 'r') as q_file:
            q_data = q_file.readlines()
        
        for line in q_data:
            data, count = self.parse_line(line)
            # check that the data is valid
            if (len(data) == 3 or len(data) == 2):
                self.q_dict[data] = count # Insert
            else:
                raise ValueError("Invald q.mle file")   

    def generate_e_dict(self, e_mle):
        """
        Process the token counts from e_mle
        """
        with open(e_mle, 'r') as e_file:
            e_data = e_file.readlines()

        for line in e_data:
            data, count = self.parse_line(line)
            if len(data) == 2:
                self.e_dict[data] = count
            else:
                raise ValueError("Invald e.mle file")   

    def predict_tag(self, prev_prev, prev, word):
        """
        Predict the tag for a word given its two previous tags
        """
        possible_tags = []
        for tag in self.tags:
            #memoization
            if (word, tag) not in self.e_cache:
                self.e_cache[(word, tag)] = self.calculate_e(word, tag)
            if (prev_prev, prev, tag) not in self.q_cache:
                self.q_cache[(prev_prev, prev, tag)] = self.calculate_q(prev_prev, prev, tag)
            tag_pr = self.e_cache[(word, tag)] * self.q_cache[(prev_prev, prev, tag)]
            possible_tags.append((tag_pr, tag))
        return max(possible_tags, key=lambda item: item[0])[1]

    def predict_line(self, words):
        """
        Predict the pos tags for each word in a sentence
        """
        predicted_tags = [START_SIGHT] * BUFFER_SIZE
        for i in xrange(0, len(words)):
            word = words[i]
            p_tag = self.predict_tag(predicted_tags[i], predicted_tags[i + 1], word)
            predicted_tags.append(p_tag)

        # Start sights
        for i in xrange(0, BUFFER_SIZE):
            predicted_tags.pop(0)
        return predicted_tags

    def format_predicted_line(self, words, tags):
        """
        Formats line for write
        """
        line_str = ""
        for i in xrange(0, len(words)):
            line_str += words[i] + SLASH + tags[i] + SPACE
        return line_str[:len(line_str) - 1] + NEW_LINE

    def predict(self, input_path):
        """
        Predicts tags for all words in input file and writes
        to outpath 
        """
        with open(input_path, 'r') as input_file:
            input_data = input_file.readlines()
        
        predicted_lines = ""
        for line in input_data:
            line = line.strip()
            # split the line into words
            line_words = line.split(SPACE)
            predicted_tokens = self.predict_line(line_words)
            predicted_lines += self.format_predicted_line(line_words, predicted_tokens)

        return predicted_lines

if __name__ == "__main__":
    if len(sys.argv) > 5:
        input_file_name = sys.argv[1]
        q_mle = sys.argv[2]
        e_mle = sys.argv[3]
        out_file_path = sys.argv[4]
        extra_file_name = sys.argv[5]

        greedy_tagger = GreedyTagger(q_mle, e_mle)
        
        formatted_predictions = greedy_tagger.predict(input_file_name)

        with open(out_file_path, 'w') as out_file:
            out_file.write(formatted_predictions)

        check_accuracy(r'C:\Users\ari-razer\Documents\biu_nlp\ass1\data\ass1-tagger-test', out_file_path)
    else:
        print "Inputs Invalid"