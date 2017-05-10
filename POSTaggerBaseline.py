# -*- coding: utf-8 -*-

import re, random, math
import numpy as np
from codecs import open
import dynet as dy
import argparse
import os
import os.path
from os.path import join
import json, codecs
from collections import OrderedDict, defaultdict

import util
import local_settings
from sefaria.model import *
from research.talmud_pos_research.language_classifier import cal_tools

def read_data(dir='', mesechta=None):
    if not dir: dir = 'data/2_matched_sefaria/json/'
    lang_dir = 'data/3_lang_tagged/json/'
    min_seg_length = 5 # min length of segment. if shorter, append the next segment with it


    all_json_files = []
    # collect all the individual filenames
    for dirpath, dirnames, filenames in os.walk(dir):
        all_json_files.extend([join(dirpath, filename) for filename in filenames if
                               filename.endswith('.json')])


    all_lang_files = []
    for dirpath, dirnames, filenames in os.walk(lang_dir):
        all_lang_files.extend([join(dirpath, filename) for filename in filenames if
                               filename.endswith('.json')])
    total_words = 0
    total_daf = 0
    total_segs = 0


    # iterate through all the files, and load them in

    segments = []
    for file,lang_file in zip(all_json_files,all_lang_files):
        if mesechta and mesechta not in file: #this is kind of hacky...but who cares?
            continue

        with open(file, 'r', encoding='utf8') as f:
            all_text = f.read()

        with open(lang_file,'r',encoding='utf8') as lf:
            all_lang_text = lf.read()
        # parse
        daf_data = json.loads(all_text)
        lang_data = json.loads(all_lang_text)

        all_words = []
        for word,lang_word in zip(daf_data['words'],lang_data):
            word_s = word['word']
            # class will be 1 if talmud, 0 if unknown
            word_known = word['class'] != 'unknown'
            word_class = 1 if lang_word['lang'] == 'aramaic' and word_known else 0
            word_lang = 1 if lang_word['lang'] == 'aramaic' else 0
            word_pos = ''
            # if the class isn't unkown
            if word_known: word_pos = word['POS']

            total_words += 1
            if word_known and word_s == u'הכא' and word_pos != u'a':
                print "OH NO! {}".format(file)
            all_words.append((word_s, word_class, word_pos, word_lang))

        total_daf += 1
        # yield it
        split_file = file.split('/')
        mesechta_name = split_file[split_file.index('json') + 1]
        daf_num = split_file[-1].split('.json')[0]
        daf = {"words": all_words, "file": '{}_{}'.format(mesechta_name, daf_num)}

        # break up daf into segments
        daf_chunk = Ref("{} {}".format(mesechta_name, daf_num)).text("he")
        ind_list, ref_list, total_len = daf_chunk.text_index_map(util.tokenize_words)

        # purposefully skip first and last seg b/c they're not necessarily syntactic
        temp_seg = None
        for i_ind in xrange(1, len(ind_list) - 1):
            if temp_seg:
                temp_seg['words'] += all_words[ind_list[i_ind]:ind_list[i_ind + 1]]
            else:
                temp_seg = {
                    "words": all_words[ind_list[i_ind]:ind_list[i_ind + 1]],
                    "file": daf['file']
                }

            if len(temp_seg['words']) >= min_seg_length:
                segments += [temp_seg]
                temp_seg = None
            total_segs += 1



    return segments


def base_line_most_probable(data):
    prec = 0.0
    num_seen = 0.0
    for daf in data:
        for w, w_class, w_pos, w_lang in daf["words"]:
            if w_class:
                num_seen += 1
                try:
                    guess = pos_hashtable[w]
                except KeyError:
                    guess = possible_pos #random.sample(possible_pos, 1)[0]
                    #print guess
                prec += int(guess == w_pos)


    print 'PREC {} ({}/{})'.format(round(100.0*prec/num_seen,3), prec, num_seen)


def make_pos_hashtable(data):
    possible_pos = defaultdict(int)
    temp_pos_hashtable = {}
    for daf in data:
        for w, w_class, w_pos, w_lang in daf["words"]:
            if w_class:
                if not w in temp_pos_hashtable:
                    temp_pos_hashtable[w] = defaultdict(int)
                temp_pos_hashtable[w][w_pos] += 1
                possible_pos[w_pos] += 1

    real_pos_hashtable = {}
    ambig_list = np.zeros(len(temp_pos_hashtable))
    counter = defaultdict(int)
    i = 0
    for w,pos_dict in temp_pos_hashtable.items():
        best_pos_list = sorted(pos_dict.items(),key=lambda x: x[1])
        ambig_list[i] = len(best_pos_list)
        real_pos_hashtable[w] = best_pos_list[0][0]
        counter[len(best_pos_list)] += 1
        i += 1

    print counter
    print 'Max {} Avg {} Var {}'.format(np.max(ambig_list), np.average(ambig_list), np.var(ambig_list))

    best_pos = sorted(possible_pos.items(), key=lambda x: x[1])[-1][0]

    return real_pos_hashtable, best_pos

all_data = read_data()
random.shuffle(all_data)
percent_training = 0.2
split_index = int(round(len(all_data) * percent_training))
train_data = all_data[split_index:]
val_data = all_data[:split_index]


pos_hashtable, possible_pos = make_pos_hashtable(train_data)
print possible_pos
base_line_most_probable(train_data)
base_line_most_probable(val_data)
pos_hashtable, possible_pos = make_pos_hashtable(all_data)
base_line_most_probable(all_data)

