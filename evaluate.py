import os
import spacy
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from HMMtagger import HMMTagger
import argparse
import pickle
import read_tags
from get_data import *
import time

def main(args): 
    start = time.time()
    nlp = spacy.load("en_core_web_sm")
   
    taggers = [pickle.load((args.hmm)[i]) for i in range(len(args.hmm))]
    
    total_right = 0
    total_size = 0 
    if args.output:
        xs = []
        ys = []
        tokens, tagged_tokens = get_tagged_tokens(args.dir)
        for words, tags in tagged_tokens: 
            # print(words)
            # print(tags)
            doc = nlp.tokenizer.tokens_from_list(list(words))
            tagger(doc)

            right = sum([spacy_token.tag_ == ref_tag 
                        for spacy_token, ref_tag in zip(doc, tags)])
            size = len(words)
            accuracy = right/size
            xs.append(size)
            ys.append(accuracy)
            total_right += right
            total_size += size
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(xs, ys, c='b', s=10)
        plt.savefig(args.output)
    else:
        for i in range(len(args.hmm)):
            tokens, tagged_tokens = get_tagged_tokens(args.dir[i])
            for words, tags in tagged_tokens: 
                # doc = nlp.tokenizer(''.join(words))
                doc=nlp.tokenizer.tokens_from_list(list(words))
                taggers[i](doc)

                right = sum([spacy_token.tag_ == ref_tag 
                            for spacy_token, ref_tag in zip(doc, tags)])
                size = len(doc)
                total_right += right
                total_size += size
            print("Accuracy: {:.2%}".format(total_right/total_size))
            print("--- %s seconds ---" % (time.time() - start))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='POS Tag, then evaluate')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True, action = "append",
                        help="Read data to tag from DIR")
    parser.add_argument("--hmm", metavar="FILE",  action = "append",
                        type=argparse.FileType('rb'), required=True,
                        help="Read hmm model from FILE")
    parser.add_argument("--universal", "-u", action="store_true", default = False,
                        help="Convert tags to universal tags")
    parser.add_argument("--output", "-o", default = False, action = "append",
                        help="Write an image to FILE")
    args = parser.parse_args()
    main(args)
 