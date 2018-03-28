import sys
import numpy as np
from ExtractFeatures import FeatureExtractor
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from config import SPACE, NEW_LINE, START_SIGHT, BUFFER_SIZE, TAB, SLASH, UNK, END, HMM_PATTERNS
# from accuracy_checker import check_accuracy  
START_SIGHT = ''

class GreedyMaxEntTagger:
    """
    A Class that calculates
    """
    def __init__(self, model, feature_map):
        self.model = model
        self.feature_extractor = FeatureExtractor()
        self.feature_map_dict = {}
        self.feature_id_map_dict = {}
        self.label_map_dict = {}
        self.vectorizer = DictVectorizer(sparse=True)
        self.process_feature_map(feature_map)
        # print self.feature_map_dict

    def process_feature_map(self, feature_map):
        """
        Process features map lines returns dictionary
        """
        f_ids = []
        for feature in feature_map:
            f, f_id = feature.split(SPACE)
            if f.isdigit():
                self.label_map_dict[int(f)] = f_id
            else:
                f_ids.append(int(f_id)) 
                self.feature_map_dict[f] = int(f_id)
                self.feature_id_map_dict[int(f_id)] = f
        f_ids.sort()
        self.vectorizer.vocabulary_ = self.feature_map_dict
        self.vectorizer.feature_names_ = [self.feature_id_map_dict[f_id] for f_id in f_ids]
       
    def prune_unknown_feats(self, feats):
        """
        Confirms that n and p strings are known features and then
        returns the correctly formatted feature strings
        """
        return {f:1 for f in feats if f in self.feature_map_dict}

    def predict_tag(self, word_context):
        """
        returns a model tag prediction given word context
        """
        word, pp_w, p_w, nn_w, n_w, pp_t, p_t = word_context
        # Get features for predicting tag
        feature_string = "form={} ".format(word) if "form={}".format(word) in self.feature_map_dict else ""
        feature_string += self.feature_extractor.apply_pattern_detectors(word) # regexs
        feature_string += self.feature_extractor.extract_fixes(word) # prefix and suffixes
        feature_string += "pp_t={}_{} p_t={} ".format(pp_t, p_t, p_t) # previous tags
        feature_string += "pp_w={} p_w={} nn_w={} n_w={} ".format(pp_w, p_w, nn_w, n_w) # previous and next words
        # format features for predicting tag
        features = feature_string.split(SPACE)
        pruned_features = self.prune_unknown_feats(features)
        # print pruned_features
        liblin_x = self.vectorizer.transform(pruned_features)        
        # print liblin_x
        tag = self.label_map_dict[int(self.model.predict(liblin_x))] # predict_tag
        # print tag
        return tag

    def predict_line(self, sentence):
        """
        Predict the pos tags for each word in a sentence
        """
        n = len(sentence)
        pp_w = p_w = ""
        nn_w = sentence[2] if n > 3 else ""
        n_w  = sentence[1] if n > 2 else ""
        pp_t = p_t = START_SIGHT 
        line_tags = []
        for i in range(n):
            word = sentence[i]
            tag = self.predict_tag([word, pp_w, p_w, nn_w, n_w, pp_t, p_t])
            line_tags.append(tag)
            pp_t, p_t, pp_w, p_w = [p_t, tag, p_w, word]
            nn_w = sentence[i+2] if i+2 < n else ""
            n_w = sentence[i+1]  if i+1 < n else ""
        return line_tags

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
        model_file_name = sys.argv[2]
        feature_map_file = sys.argv[3]
        out_file_path = sys.argv[4]
        extra_file_name = sys.argv[5]

        model = joblib.load(model_file_name) 
        feature_map = file(feature_map_file).read().splitlines()
        greedy_tagger = GreedyMaxEntTagger(model, feature_map)
        formatted_predictions = greedy_tagger.predict(input_file_name)

        with open(out_file_path, 'w') as out_file:
            out_file.write(formatted_predictions)

        check_accuracy(r'C:\Users\ari-razer\Documents\biu_nlp\ass1\data\ass1-tagger-test', out_file_path)
    else:
        print "Inputs Invalid"