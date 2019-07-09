import pyLDAvis
import pandas as pd
import numpy as np
import json
import sys
sys.path.append("..")
from collections import Counter

'''
Pass a directory
Output html visualization
'''

def load_topic_word(path):
    topic_word_df = pd.read_csv(path+'topic_word_matrix.csv', header=None)
    topic_word = topic_word_df.to_numpy(copy=True)
    return topic_word

def load_doc_topic(path):
    with open(path+'doc_topic_matrix.txt', 'r') as f:
        doc_topic_dist = f.read()
    d1 = doc_topic_dist.replace('WrappedArray', '')
    d2 = d1.replace('(', '')
    d3 = d2.replace(')', '')
    # store for ease of use later
    with open('../data/doc_topic_matrix_parsed.csv', 'w') as f:
        f.write(d3)
    doc_top_df = pd.read_csv('../data/doc_topic_matrix_parsed.csv', header=None)
    doc_top = doc_top_df.to_numpy(copy=True)
    return doc_top

def load_doc_lengths(path):
    # document lengths
    doc_lengths = np.loadtxt(path+'data/doc_lengths.txt')
    return doc_lengths

def load_vocab(path):
    with open('../data/vocabulary.txt', 'r') as f:
        vocab = f.read()
    vocabulary = vocab.split('\n')
    vocabulary.pop()
    return vocabulary

def load_term_freq(path):
    with open(path+'data/tfdata.json') as f:
        tf = f.read
    tf2 = tf.replace('\n', ',')
    header = "{\"tfdata\": ["
    footer = "]}"
    tf3 = "".join([header,tf2])
    tf4 = "".join([tf3,footer])
    tfjson = json.loads(tf4)
    term_freq_corpus = term_freq(tfjson)
    return term_freq_corpus

def term_freq(json_doc):
    c = Counter(dict(zip(json_doc['tfdata'][0]['indices'], json_doc['tfdata'][0]['values'])))
    for i in range(1, len(json_doc['tfdata'])):
        c.update(dict(zip(json_doc['tfdata'][i]['indices'], json_doc['tfdata'][i]['values'])))
    out = []
    for k in sorted(c.keys()):
        out.append(c[k])
        if i % 100 == 0:
            print('Iteration: ' + str(i))
    return out


# write out the file

#########################################################

if __name__ == '__main__':
    path = sys.argv[1]
    topic_word = load_topic_word(path)
    doc_topic = load_doc_topic(path)
    doc_lengths = load_doc_lengths(path)
    vocab = load_vocab(path)
    term_freq = load_term_freq(path)



