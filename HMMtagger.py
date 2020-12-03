#!/usr/bin/env python3

import sys
from numpy import argmax, zeros, array, float32, ones, zeros, log
import spacy
import os
from collections import defaultdict, Counter
import argparse
import pickle
# from get_data import *
from get_data import *
import time


class HMMTagger():
    def __init__(self, nlp, alpha=0.1, vocabsize = None):
        self.tags =  ["<<START>>",]+ tagSet
        self.vocab = ["<<OOV>>",]
        self.alpha = alpha
        self.vocabsize = vocabsize

    """Given an input string, return the lower case of the string if it is present
       in vocab, otherwise return <<OOV>>
    """
    def clean_token(self, text):
        return text.lower() if text in self.vocab else "<<OOV>>"

    """Given a word, if the word is present in vocab, return the index.
        otherwise, return 0"""
    def word_to_index(self, w):
        return self.vocab.index(self.clean_token(w))

    """Given a tag, return the index of the tag in tags"""
    def tag_to_index(self, t):
        try:
            return self.tags.index(t)
        except:
            return len(self.tags)-1

    """Get the costs of all the possible starting tags"""
    def get_start_costs(self):
        return self.tag_tag_probs[self.tag_to_index("<<START>>"),:]

    """Get the costs of the tags for a given token"""
    def get_token_costs(self, token, tag_j = None):
        return self.tag_word_probs[:,self.word_to_index(token.text)]

    """Given a dirctory, read all the files as string, and record all unique lower-cased
        words in vocab"""
    def update_vocab(self, vocabList):
        token_counter = Counter(vocabList)
        vocabSet = []
        if self.vocabsize:
            most_common = token_counter.most_common(self.vocabsize)
            vocabSet = [key for key,_ in most_common]
        else:
            vocabSet= list(token_counter.keys())
        self.vocab += vocabSet

    """Given a list of words and tags, record the frequencies of the word-tag pairs, and
        the transitions from tags to tags in the matrices of tag_word and tag_tag accordingly"""
    def do_train_sent(self, tagged_tokens): 
        # set prev_tag to 0
        prev_tag = self.tag_to_index("<<START>>")
        # iterating through all the words and the corresponding tags
        for tagged_words in tagged_tokens:      # for a word and its tag
            (word, tag) = tagged_words
            t_i = self.tag_to_index(tag)        # t_i = the index of the tag
            w_i = self.word_to_index(word)      # w_i is the index of the word
            self.tag_word[t_i][w_i] += 1        # add 1 to the count of word and
                                                # tag pair in tag_word
                                                # (how likely is this word
                                                # tagged with this tag)
            self.tag_tag[prev_tag][t_i] += 1    # add 1 to the count of previous
                                                # tag and this tag pair in tag tag
                                                # (how likely does this tag
                                                # follow the previous tag)
            prev_tag = t_i                      # update previous tag 

    """Given a dictory, """
    def train(self, train_dir):
        # record all the words in self.vocab
        tokens, tagged_tokens = get_tagged_tokens(train_dir)
        self.update_vocab(tokens)
        # initialize the tag_word matrix (|words|*|tags|) with every entry to be alpha
        # row -> words in vocab
        # col -> tags 
        self.tag_word  = ones((len(self.tags), len(self.vocab))) * self.alpha
        # initialize the tag_tag matrix (|tags|*|tags|) with every entry to be alpha
        # row and col -> tags
        self.tag_tag =  ones((len(self.tags), len(self.tags))) * self.alpha
        # fill in the entries
        for words, tags in tagged_tokens: # for a lists of words and their tags
            self.do_train_sent(zip(words,tags))      # record frequencies of word-tag pairs
                                                 # and the transitions probability
                                                 # between tags and predecessor tags
        # normalize both matrices
        self.normalize_probabilities()          

    """Normalize a matrix
        ex: input: [[a b],
                   [c d]]
                   
            output: [[log(1/c) log(b/ac)],
                    [log(c/bd) log(1/b)]] """
    def normalize(self, m):
        return (log(m).transpose() - log(m.sum(axis=1))).transpose()
        
    """Normalize tag_word and tag_tag"""
    def normalize_probabilities(self):
        # normalize tag_word and set it to tag_word_probs
        self.tag_word_probs = self.normalize(self.tag_word)
        # normalize tag_tag and set it to tag_tag_probs
        self.tag_tag_probs = self.normalize(self.tag_tag)

    """Predict the tags of the tokens"""
    def __call__(self, tokens):
        self.predict(tokens)
        
    """Predict the tags of the tokens"""
    def predict(self, tokens):
        # Build DP table, which should be |sent| x |tags|
        cost_table = zeros((len(tokens), len(self.tags)), float32)
        bt_table   = zeros((len(tokens), len(self.tags)), int)
        if len(tokens) != 0:
            for token_i, token in enumerate(tokens):        # for a token and its index in tokens
                token_costs = self.get_token_costs(token)   # token_costs = all possible word-tag pair
                                                            #   for the current token
                if token_i == 0:                            # if it is the first token in input token list
                    prev_costs = self.get_start_costs() + token_costs
                                                            # initial cost is lists of all possible
                                                            #   starting tags plus the costs for the current words
                    bt_table[token_i, :] = -1               # set all entries in the first row of bt_table to -1
                else:                                       # if the token is not the starting token
                    costs = self.tag_tag_probs.copy()       # make a copy of tag_tag_probs
                    costs = (costs.transpose() + prev_costs).transpose() # same to every column
                                                            # for every previous cost in each tag, 
                                                            #   add the previous cost
                    costs += token_costs                    # for previous cost in each tag, 
                                                            #   add the token cost to each entry in the column
                    
                    prev_costs = costs.max(axis=0)          # for each tag, take the highest cost across all tag-tag pairs
                    bt_table[token_i,:] = costs.argmax(axis=0) # update bt_table, so that every entry in the row holds
                                                            #       the indexes of the highest-cost tag-tag pairs
                cost_table[token_i, :] = prev_costs         # update the cost_table 

            # Find the highest-probability tag for last word
            best_last_tag = argmax(prev_costs)

            # Trace back through the bt_table
            self.backtrace(bt_table, tokens, best_last_tag)

    # Given a backtrack table, and a list of tokens, and the best last tage
    # adding attribute tag_ to every element in the list of tokens
    def backtrace(self, bt_table, tokens, best_last_tag):
        current_row = len(tokens)-1
        for t in list(tokens)[::-1]:                # start at the end of the tokens
            t.tag_ = self.tags[best_last_tag]       # set tag_ to be the tag based on index of the best tag
            best_last_tag = bt_table[current_row, best_last_tag]  # choose the next best tag in the previous row(tag)
            current_row -= 1

def main(args):
    start = time.time()
    nlp = spacy.load("en_core_web_sm")
    tagger = HMMTagger(nlp, alpha=args.alpha, vocabsize = args.vocabsize)
    # for i, train_dir in enumerate(args.dir):
    tagger.train(args.dir)
    pickle.dump(tagger, args.output[0])
    print("--- %s seconds ---" % (time.time() - start))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Train (and save) hmm models for POS tagging')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True, action="append",
                        help="Read training data from DIR")
    parser.add_argument("--output", "-o", metavar="FILE", 
                        type=argparse.FileType('wb'), required=True, action="append",
                        help="Save output to FILE")
    parser.add_argument("--alpha", "-a", default=0.1, 
                        help="Alpha value for add-alpha smoothing")
    parser.add_argument("--vocabsize", "-v", default = None, type=int,
                        help="Convert tags to universal tags")

    args = parser.parse_args()
    main(args)