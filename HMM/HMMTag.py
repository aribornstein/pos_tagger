import sys
import collections
import numpy as np
from config import SPACE, NEW_LINE, START_SIGHT, BUFFER_SIZE, TAB, SLASH, UNK, END, HMM_PATTERNS
from accuracy_checker import check_accuracy  

class HMMTagger:
    """
    A Class that calculates e and q for HMM models
    """
    def __init__(self, q_mle, e_mle):
        self.tags = set()
        self.q_dict = {} # q memoization cache
        self.q_cache = {}
        self.e_dict = {}
        self.e_cache = {}  # e memoization cache
        self.known_words_set = set()
        self.tag_count_dict = {}
        self.total_word_count = None
            
        # read the q and e tables to local dicts
        self.generate_q_dict(q_mle)
        self.generate_e_dict(e_mle)
        self.pruned_word_tags = {word:set([]) for word,token in self.e_dict}
        self.generate_possible_word_tags()

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
        self.tags = set(self.tag_count_dict.keys()) - set([START_SIGHT, END])
        self.known_words_set.update([word_tag_tuple[0] for word_tag_tuple in self.e_dict.keys()])

    def generate_possible_word_tags(self):
        """
        generates dictionary of possible tags for each word 
        needed for pruning
        """
        for word, tag in self.e_dict:
            self.pruned_word_tags[word].add(tag)

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
        if tag != START_SIGHT and prev != END:
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

    def prune_tags(self, word):
        if word == END:
            return {END}
        prunded_tags = self.pruned_word_tags[word] if word in self.pruned_word_tags else self.tags
        return prunded_tags - set([START_SIGHT, END])
        
    def predict_line(self, words):
        """
        Predict the pos tags for each word in a sentence
        """
        result = []
        words.append(END)
        n = len(words)
        best_score = {(0,START_SIGHT, START_SIGHT):0.0}
        backpointer = {(0, START_SIGHT):None}

        # pruning
        possible_pp_tags = None
        possible_prev_tags = None
        possible_next_tags = self.prune_tags(words[0])

        # Start step
        for next_tag in possible_next_tags:
            self.update_vertirbi(0, START_SIGHT, START_SIGHT, next_tag, words[0], best_score, backpointer)
        possible_prev_tags = self.prune_tags(words[0])
        possible_next_tags = self.prune_tags(words[1])             
        for prev in possible_prev_tags:
            for next_tag in possible_next_tags:
                self.update_vertirbi(1, START_SIGHT, prev, next_tag, words[1], best_score, backpointer)    
        if n > 2:
            # Middle step
            for i in xrange(2, n-1):
                possible_pp_tags = self.prune_tags(words[i-2])
                possible_prev_tags = self.prune_tags(words[i-1])
                possible_next_tags = self.prune_tags(words[i])                     
                for prev_prev in possible_pp_tags:
                    for prev in possible_prev_tags:
                        for next_tag in possible_next_tags:
                            self.update_vertirbi(i, prev_prev, prev, next_tag, words[i], best_score, backpointer)

            # End step http://www.dictionary.com/browse/penultimate
            possible_pp_tags = self.prune_tags(words[n-3])
            possible_prev_tags = self.prune_tags(words[n-2])        
            for antepenultimate in possible_pp_tags:
                for penultimate in possible_prev_tags:
                    self.update_vertirbi(n-1, antepenultimate, penultimate, END, words[n-1], best_score, backpointer)

        # Back point
        next_tag = backpointer[(n, END)]

        while next_tag != None:
            next_tag = next_tag[1]
            result.append(next_tag[1])
            next_tag = backpointer[next_tag]
        result.pop() # pop off START

        return list(reversed(result))


    def update_vertirbi(self, i, prev_prev, prev, tag, word, best_score, backpointer):
        """
        Calculates verterbi scores for index i and updates tables if needed
        """
        # update memoization cache
        if (word, tag) not in self.e_cache:
            self.e_cache[(word, tag)] = self.calculate_e(word, tag)
        if (prev_prev, prev, tag) not in self.q_cache:
            self.q_cache[(prev_prev, prev, tag)] = self.calculate_q(prev_prev, prev, tag)
        #retrieve e and q values from cache and calculate score
        q = self.q_cache[(prev_prev, prev, tag)]
        e = self.e_cache[(word, tag)]
        score = best_score[(i, prev_prev, prev)] + np.log(q) + np.log(e)
        # Update best score dict        
        if (not (i+1, prev, tag) in best_score) or score > best_score[(i+1, prev, tag)]:
            best_score[(i+1, prev, tag)] = score
        # Update backpointer (optimasetd)
        if (not (i + 1, tag) in backpointer) or (score > backpointer[(i+1, tag)][0]):
            backpointer[(i+1, tag)] = (score, (i, prev))              

    def format_predicted_line(self, words, tags):
        """
        Formats line for write
        """
        line_str = ""
        for i in xrange(0, len(words)):
            line_str += words[i] + SLASH + tags[i] + SPACE
        print line_str[:len(line_str) - 1] + NEW_LINE
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
            print "line words", line_words
            predicted_tokens = self.predict_line(line_words)
            line_words.pop() # remove END
            predicted_lines += self.format_predicted_line(line_words, predicted_tokens)

        return predicted_lines

if __name__ == "__main__":
    if len(sys.argv) > 5:
        input_file_name = sys.argv[1]
        q_mle = sys.argv[2]
        e_mle = sys.argv[3]
        out_file_path = sys.argv[4]
        extra_file_name = sys.argv[5]

        hmm_tagger = HMMTagger(q_mle, e_mle)
        
        formatted_predictions = hmm_tagger.predict(input_file_name)

        with open(out_file_path, 'w') as out_file:
            out_file.write(formatted_predictions)

        check_accuracy(r'C:\Users\ari-razer\Documents\biu_nlp\ass1\ner\dev', out_file_path)

        # check_accuracy(r'C:\Users\ari-razer\Documents\biu_nlp\ass1\data\ass1-tagger-test', out_file_path)
    else:
        print "Inputs Invalid"