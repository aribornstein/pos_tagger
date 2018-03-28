import sys
import numpy as np
from ExtractFeatures import FeatureExtractor
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from config import SPACE, NEW_LINE, BUFFER_SIZE, START_SIGHT, TAB, SLASH, UNK, END
from accuracy_checker import check_accuracy  

class MEMMTagger:
    """
    A MEMM tagger
    """
    def __init__(self, model, feature_map, word_tag_map):
        self.tags = None
        self.pruned_word_tags = {}
        self.word_cache = {}
        self.model = model
        self.feature_extractor = FeatureExtractor()
        self.feature_map_dict = {}
        self.feature_id_map_dict = {}
        self.label_map_dict = {}
        self.label_id_map_dict = {}
        self.vectorizer = DictVectorizer(sparse=True)
        self.process_feature_map(feature_map)
        self.word_tag_map = {}
        self.generate_word_tag_map(word_tag_map)
        self.end_id = self.label_id_map_dict[END]
        self.pruned_word_tags = {word:set([]) for word,token in self.word_tag_map}
        self.generate_possible_word_tags()
        print [(self.label_map_dict[int(c)],c) for c in self.model.classes_]

    def generate_word_tag_map(self, word_tag_map):
        """
        Process the token counts from e_mle
        """
        for line in word_tag_map:
            data, count = self.parse_line(line)
            if len(data) == 2:
                self.word_tag_map[data] = count
            
    def process_feature_map(self, feature_map):
        """
        Process features map lines returns dictionary
        """
        f_ids = []
        for feature in feature_map:
            f, f_id = feature.split(SPACE)
            if f.isdigit():
                self.label_map_dict[int(f)] =  f_id
                self.label_id_map_dict[f_id] = int(f)
            else:
                f_ids.append(int(f_id)) 
                self.feature_map_dict[f] = int(f_id)
                self.feature_id_map_dict[int(f_id)] = f
        f_ids.sort()
        self.tags = self.label_map_dict.keys()
        self.vectorizer.vocabulary_ = self.feature_map_dict
        self.vectorizer.feature_names_ = [self.feature_id_map_dict[f_id] for f_id in f_ids]
       
    def prune_unknown_feats(self, feats):
        """
        Confirms that n and p strings are known features and then
        returns the correctly formatted feature strings
        """
        return {f:1 for f in feats if f in self.feature_map_dict}

    def generate_possible_word_tags(self):
        """
        generates dictionary of possible tags for each word 
        needed for pruning
        """
        for word, tag in self.word_tag_map:
            if tag != START_SIGHT: # find more elegant way of handling the END in e.mle
                self.pruned_word_tags[word].add(self.label_id_map_dict[tag])

    def parse_line(self, line):
        """
        Parses a line of tokens
        """
        data, count = line.rsplit(TAB, 1)
        data = data.split(SPACE)
        return tuple(data), float(count)

    def prune_tags(self, word):
        if word == END:
            return {self.end_id}
        prunded_tags = self.pruned_word_tags[word] if word in self.pruned_word_tags else self.tags
        return set(prunded_tags) - {self.end_id}

    def predict_tag_score(self, word_context, tag):
        """
        returns a model tag score  given word context
        """
        word, pp_w, p_w, nn_w, n_w, pp_t, p_t = word_context
        pp_t = self.label_map_dict[pp_t] if pp_t != START_SIGHT else START_SIGHT
        p_t = self.label_map_dict[p_t] if p_t != START_SIGHT else START_SIGHT

        # Get features for predicting tag
        feature_string = "form={} ".format(word) if "form={}".format(word) in self.feature_map_dict else ""
        feature_string += self.feature_extractor.apply_pattern_detectors(word) # regexs
        feature_string += self.feature_extractor.extract_fixes(word) # prefix and suffixes
        feature_string += "pp_t={}_{} p_t={} ".format(pp_t, p_t, p_t) # previous tags
        feature_string += "pp_w={} p_w={} nn_w={} n_w={} ".format(pp_w, p_w, nn_w, n_w) # previous and next words
        # format features for predicting tag
        features = feature_string.split(SPACE)
        pruned_features = self.prune_unknown_feats(features)
        liblin_x = self.vectorizer.transform(pruned_features)
        return self.model.predict_log_proba(liblin_x)[0][tag]
        
    def create_wc(self, i, words, pp_t, p_t):
        """
        Creates a word context [word, pp_w, p_w, nn_w, n_w, pp_t, p_t]
        """
        n_word  =  words[i+1] if (len(words) > i+1) else ""
        nn_word =  words[i+2] if (len(words) > i+2) else ""
        p_word  =  words[i-1] if (i > 0) else ""
        pp_word =  words[i-2] if (i > 1) else ""

        return [words[i], pp_word, p_word, nn_word, n_word, pp_t, p_t]

    def predict_line(self, words):
        """
        Predict the pos tags for each word in a sentence
        """
        result = []
        words.append(END)
        n = len(words)
        best_score = {(0, START_SIGHT, START_SIGHT):0.0}
        backpointer = {(0, START_SIGHT):None}

        # pruning
        possible_pp_tags = None
        possible_prev_tags = None
        possible_next_tags = self.prune_tags(words[0])

        # Start step
        for next_tag in possible_next_tags:
            word_context = self.create_wc(0, words, START_SIGHT, START_SIGHT)
            self.update_vertirbi(0, next_tag, word_context, best_score, backpointer)

        possible_prev_tags = self.prune_tags(words[0])
        possible_next_tags = self.prune_tags(words[1])             
        for prev in possible_prev_tags:
            for next_tag in possible_next_tags:
                word_context = self.create_wc(1, words, START_SIGHT, prev)
                self.update_vertirbi(1, next_tag, word_context, best_score, backpointer)    
        if n > 2:    
            # Middle step
            for i in xrange(2, n-1):
                possible_pp_tags = self.prune_tags(words[i-2])
                possible_prev_tags = self.prune_tags(words[i-1])
                possible_next_tags = self.prune_tags(words[i])                     
                for prev_prev in possible_pp_tags:
                    for prev in possible_prev_tags:
                        for next_tag in possible_next_tags:
                            word_context = self.create_wc(i, words, prev_prev, prev)
                            self.update_vertirbi(i, next_tag, word_context, best_score, backpointer)
            # End step http://www.dictionary.com/browse/penultimate
            possible_pp_tags = self.prune_tags(words[n-3])
            possible_prev_tags = self.prune_tags(words[n-2])        
            for antepenultimate in possible_pp_tags:
                for penultimate in possible_prev_tags:
                    word_context = self.create_wc(n-1, words, antepenultimate, penultimate)
                    self.update_vertirbi(n-1, self.end_id, word_context, best_score, backpointer)

        # Back point
        next_tag = backpointer[(n, self.end_id)]

        # print next_tag
        while next_tag != None:
            next_tag = next_tag[1]
            result.append(next_tag[1])
            next_tag = backpointer[next_tag]
        result.pop() # pop off START

        return list(reversed(result))


    def update_vertirbi(self, i, tag, word_context, best_score, backpointer):
        """
        Calculates verterbi scores for index i and updates tables if needed
        """
        word, pp_w, p_w, nn_w, n_w, pp_t, p_t = word_context
        
        # update memoization cache
        if (tuple(word_context), tag) not in self.word_cache:
            self.word_cache[(tuple(word_context), tag)] = self.predict_tag_score(word_context, tag)

        score = best_score[(i, pp_t, p_t)] + self.word_cache[(tuple(word_context), tag)]
        # Update best score dict        
        if (not (i+1, p_t, tag) in best_score) or score > best_score[(i+1, p_t, tag)]:
            best_score[(i+1, p_t, tag)] = score
        # Update backpointer (optimasetd)
        if (not (i + 1, tag) in backpointer) or (score > backpointer[(i+1, tag)][0]):
            backpointer[(i+1, tag)] = (score, (i, p_t))              

    def format_predicted_line(self, words, tags):
        """
        Formats line for write
        """
        line_str = ""
        for i in xrange(len(words)):
            line_str += words[i] + SLASH + self.label_map_dict[tags[i]] + SPACE
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
            predicted_tokens = self.predict_line(line_words)
            line_words.pop() # remove END
            predicted_lines += self.format_predicted_line(line_words, predicted_tokens)

        return predicted_lines

if __name__ == "__main__":
    if len(sys.argv) > 5:
        input_file_name = sys.argv[1]
        modelname = sys.argv[2]
        feature_map_file = sys.argv[3]
        word_tag_map_file = sys.argv[4]
        out_file_name = sys.argv[5]

        model = joblib.load(modelname) 
        feature_map = file(feature_map_file).read().splitlines()
        word_tag_map = file(word_tag_map_file).read().splitlines()
        memm_tagger = MEMMTagger(model, feature_map, word_tag_map)
        
        formatted_predictions = memm_tagger.predict(input_file_name)

        with open(out_file_name, 'w') as out_file:
            out_file.write(formatted_predictions)

        # check_accuracy(r'C:\Users\ari-razer\Documents\biu_nlp\ass1\data\ass1-tagger-test', out_file_name)
        check_accuracy(r'C:\Users\ari-razer\Documents\biu_nlp\ass1\ner\dev', out_file_name)

    else:
        print "Inputs Invalid"