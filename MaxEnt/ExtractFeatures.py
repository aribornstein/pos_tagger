import sys
import collections
from config import SPACE, NEW_LINE, TAB, SLASH, END, START_SIGHT, FEATURE_PATTERNS

class FeatureExtractor:
    """
    A Class that extracts features from texxt
    """
    def __init__(self, corpus=None):
        self.pattern_cache = {}
        self.corpus = corpus
        self.rare_words = {}
        
    def generate_vocab(self, pruning_threshold):
        """
        Generates vocab prunes words that appear less than the pruning threshold
        """
        words = []
        for sentence in self.corpus:
            sentence = sentence.strip()
            tokens = sentence.split(SPACE)
            words += [token.rsplit(SLASH, 1)[0] for token in tokens]
        word_count = collections.Counter(words)
        self.rare_words = set([word for word in word_count if word_count[word] < pruning_threshold])
        print "Rare Words ", len(self.rare_words)

    def get_features(self):
        """
        Returns a string of formated features for 
        writing to file
        """
        features = []
        for sentence in self.corpus:
            features += self.extract_sentence_features(sentence)
            
        return NEW_LINE.join(features)

    def extract_sentence_features(self, sentence):
        """
        Extract all the line_features from a sentnece
        """
        line_features = []
        sentence = sentence.strip()
        tokens = sentence.split(SPACE)
        tokens.append("{}{}{}".format(END, SLASH, END))

        n = len(tokens)
        pp_w = p_w = ""
        nn_w = tokens[2].rsplit(SLASH, 1)[0] if n > 3 else ""
        n_w  = tokens[1].rsplit(SLASH, 1)[0] if n > 2 else ""
        pp_t = p_t = START_SIGHT 
        for i in range(n):
            word, tag = tokens[i].rsplit(SLASH, 1)
            feature = "{} form={} ".format(tag, word)
            feature += self.apply_pattern_detectors(word) # regexs
            feature += self.extract_fixes(word) # prefix and suffixes
            feature += "pp_t={}_{} p_t={} ".format(pp_t, p_t, p_t) # previous tags
            feature += "pp_w={} p_w={} nn_w={} n_w={} ".format(pp_w, p_w, nn_w, n_w) # previous and next words
            # feature += "len={} ".format(len(word))
            line_features.append(feature)
            pp_t, p_t, pp_w, p_w = [p_t, tag, p_w, word]
            nn_w = tokens[i+2].rsplit(SLASH, 1)[0] if i+2 < n else ""
            n_w = tokens[i+1].rsplit(SLASH, 1)[0]  if i+1 < n else ""
        return line_features

    def extract_fixes(self, word):
        """
        Extracts sufixes and prefixes of n 2 and 3
        """
        if not word.isdigit():
                return "prefix_1={} prefix_2={} prefix_3={} prefix_4={} suffix_1={} suffix_2={} suffix_3={} suffix_4={} "\
                        .format(word[0],word[:2], word[:3], word[:4], word[-1:], word[-2:],word[-3:], word[-4:])
        
        return "prefix_1= prefix_2= prefix_3= prefix_4= suffix_1= suffix_2= suffix_3= suffix_4= "
        
    def apply_pattern_detectors(self, word):
        """
        Extracts regular expression features with known rules
        """
        if word in self.pattern_cache:
            return self.pattern_cache[word] # cache to speed up

        pattern_features = ""
        for p in FEATURE_PATTERNS:
            feature_found = bool(p[0].match(word))
            pattern_features += "{}={} ".format(p[1], feature_found)
        
        self.pattern_cache[word] = pattern_features
        return pattern_features

if __name__ == "__main__":
    if len(sys.argv) > 2:
        corpus_file = sys.argv[1]
        features_file = sys.argv[2]

        corpus = file(corpus_file).readlines()
        feature_extractor = FeatureExtractor(corpus)
        feature_extractor.generate_vocab(pruning_threshold=10)
        formatted_feature = feature_extractor.get_features()

        with open(features_file, 'w') as out_file:
            out_file.write(formatted_feature)

    else:
        print "Inputs Invalid"