# -*- coding: utf-8 -*-

import re,codecs,json,os,random
from collections import OrderedDict
from os import listdir
from os.path import isfile, join

import util
import local_settings
from research.talmud_pos_research.language_classifier import cal_tools
from sefaria.model import *
from sefaria.utils import hebrew

random.seed(2823274491)

def make_aramaic_training():
    abbrev_dict = json.load(codecs.open('data/1_cal_input/abbreviations.json',encoding='utf8'))
    for abbrev, defs in abbrev_dict.items():
        sorted_defs = sorted(defs.items(),key=lambda x: x[1])
        abbrev_dict[abbrev] = sorted_defs[0][0]


    training = []
    num_found = 0
    num_missed = 0
    with open('data/1_cal_input/caldbfull.txt','rb') as cal:
        for line in cal:
            line_obj = cal_tools.parseCalLine(line,True,withshinsin=False)
            temp_word = line_obj['word']

            words = []
            if u"'" in temp_word:
                temp_word = temp_word.replace(u"'",u'')
                if temp_word in abbrev_dict:
                    words = re.split(ur'\s+',abbrev_dict[temp_word])
                    num_found += 1
                else:
                    num_missed += 1
                    #print u'missed {}'.format(temp_word)
            else:
                words = [temp_word]

            for w in words:
                training.append({'word':w,'tag':'aramaic'})

    print u'Num abbrevs replaced {}. Num Missed {}'.format(num_found,num_missed)
    return training

def make_mishnaic_training():
    training = []
    num_mishnah_per_mesechta = 30000  # effectively all mishnah
    mishnah_indexes = [library.get_index(ind) for ind in library.get_indexes_in_category("Mishnah")]

    mishnah_indexes += [library.get_index(ind) for ind in library.get_indexes_in_category("Torah")]

    mish_set = set()
    num_removed = 0
    for ind in mishnah_indexes:
        mishna_segs = ind.all_section_refs()
        if len(mishna_segs) >= num_mishnah_per_mesechta:
            mishna_segs = mishna_segs[:num_mishnah_per_mesechta]
        for seg in mishna_segs:
            first_sec_str = hebrew.strip_cantillation(seg.text('he').as_string(), strip_vowels=True)
            word_list = util.tokenize_words(first_sec_str)
            for word in word_list:
                if random.random() > 0.45 and word in mish_set:
                    num_removed += 1
                    continue
                training.append({'word':word,'tag':'mishnaic'})
                mish_set.add(word)
    print "Num Mishna removed: {}".format(num_removed)
    return training

def merge_sets(a,m):
    a_set = {}
    m_set = {}

    let_set = set()
    for w in a:
        for c in w['word']:
            let_set.add(c)





    for w in m:
        if w['word'] not in m_set:
            m_set[w['word']] = 0
        m_set[w['word']] +=  1


    num_in_m = 0
    """
    with open('data/1_cal_input/jbaforms.txt','rb') as jba:
        for line in jba:
            line_obj = cal_tools.parseJBALine(line,True,withshinsin=False)
            if 'word' in line_obj and line_obj['word'] not in m_set: # don't add ambiguous words from jba. there are too many
                bad_char = False
                for c in line_obj['word']:
                    if c not in let_set:
                        print 'continued'
                        bad_char = True
                        break
                if not bad_char:
                    a.append({'word':line_obj['word'],'tag':'aramaic'})
            elif 'word' in line_obj and line_obj['word'] in m_set:
                num_in_m += 1
    """
    for w in a:
        if w['word'] not in a_set:
            a_set[w['word']] = 0
        a_set[w['word']] += 1


    ambig_set = set()
    for word,count in a_set.items():
        if word in m_set and count < 10*m_set[word] and m_set[word] < 10*count:
            ambig_set.add(word)



    ambig = []
    a_merge = []
    m_merge = []
    num_deleted = 0
    for w in a:
        if w['word'] in ambig_set:
            num_deleted += 1
            ambig.append({'word':w['word'],'tag':'ambiguous'})
        else:
            a_merge.append(w)

    for w in m:
        if w['word'] in ambig_set:
            num_deleted += 1
            ambig.append({'word':w['word'],'tag':'ambiguous'})
        else:
            m_merge.append(w)

    print num_deleted
    print "NUM IN M {}".format(num_in_m)
    print 'YO A {} M {}'.format(len(a),len(m))
    return a_merge + m_merge + ambig,len(a_merge),len(m_merge),len(ambig)



def print_tagged_corpus_to_html_table(lang_out):
    str = u"<html><head><style>h1{text-align:center;background:grey}td{text-align:center}table{margin-top:20px;margin-bottom:20px;margin-right:auto;margin-left:auto;width:1200px}.aramaic{background-color:blue;color:white}.mishnaic{background-color:red;color:white}.ambiguous{background-color:yellow;color:black}</style><meta charset='utf-8'></head><body>"
    for daf in lang_out:
        str += u"<h1>DAF {}</h1>".format(daf)
        str += u"<table>"
        count = 0
        while count < len(lang_out[daf]):
            row_obj = lang_out[daf][count:count+10]
            row = u"<tr>"
            for w in reversed(row_obj):
                row += u"<td class='{}'>{}</td>".format(w['lang'],w['word'])
            row += u"</tr>"
            #row_sef += u"<td>({}-{})</td></tr>".format(count,count+len(row_obj)-1)
            str += row
            count += 10
        str += u"</table>"
        str += u"</body></html>"
    return str


def dilate_lang():
    lang_tagged_path = 'data/3_lang_tagged'
    lang_tagged_dilated_path = 'data/4_lang_tagged_dilated'
    mesechtot_names = ['Berakhot','Shabbat','Eruvin','Pesachim','Bava Kamma','Bava Metzia','Bava Batra']
    for mesechta in mesechtot_names:
        util.make_folder_if_need_be('{}/json/{}'.format(lang_tagged_path, mesechta))
        mesechta_path = '{}/json/{}'.format(lang_tagged_path, mesechta)

        def sortdaf(fname):
            daf = fname.split('/')[-1].split('.json')[0]
            daf_int = int(daf[:-1])
            amud_int = 1 if daf[-1] == 'b' else 0
            return daf_int*2 + amud_int

        files = [f for f in listdir(mesechta_path) if isfile(join(mesechta_path, f))]
        files.sort(key=sortdaf)
        html_out = OrderedDict()
        for i_f,f_name in enumerate(files):
            lang_out = []
            lang_in = json.load(codecs.open('{}/{}'.format(mesechta_path,f_name), "rb", encoding="utf-8"))
            for i_w,w in enumerate(lang_in):
                if 1 < i_w < len(lang_in)-1:
                    neigh = [lang_in[i_w-1]['confidence'],lang_in[i_w+1]['confidence']]
                elif i_w < len(lang_in) - 1:
                    neigh = [lang_in[i_w+1]['confidence']]
                else:
                    neigh = [lang_in[i_w-1]['confidence']]
                neigh_conf = [sum([c[0] for c in neigh])/2,sum([c[1] for c in neigh])/2]

                weight = 1.1
                new_conf = [sum([neigh_conf[0],weight*w['confidence'][0]]),sum([neigh_conf[1],weight*w['confidence'][1]])]
                new_lang = 'aramaic' if new_conf[0] > new_conf[1] else 'mishnaic'
                lang_out.append({'word':w['word'],'lang':new_lang,'confidence':new_conf})

            util.make_folder_if_need_be("{}/json/{}".format(lang_tagged_dilated_path,mesechta))
            fp = codecs.open("{}/json/{}/{}".format(lang_tagged_dilated_path,mesechta,f_name), "wb", encoding='utf-8')
            json.dump(lang_out, fp, indent=4, encoding='utf-8', ensure_ascii=False)
            fp.close()

            daf = f_name.split('/')[-1].split('.json')[0]
            html_out[daf] = lang_out
            if i_f % 10 == 0:
                print '{}/{}'.format(mesechta,f_name)
                html = print_tagged_corpus_to_html_table(html_out)
                util.make_folder_if_need_be("{}/html/{}".format(lang_tagged_dilated_path, mesechta))
                fp = codecs.open("{}/html/{}/{}.html".format(lang_tagged_dilated_path, mesechta, daf), "wb",
                                 encoding='utf-8')
                fp.write(html)
                fp.close()
                html_out = OrderedDict()

#CONTEXT

def make_aramaic_training_context():
    training = []
    with open('data/1_cal_input/caldbfull.txt','rb') as cal:
        temp_phrase = []
        curr_line_num = None
        curr_word_num = None
        for line in cal:
            try:
                lineObj = cal_tools.parseCalLine(line,True,False)
            except IndexError:
                continue
            if curr_line_num is None:
                curr_line_num = lineObj['line_num']
            if curr_word_num is None:
                curr_word_num = lineObj['word_num'] - 1
            if curr_line_num == lineObj['line_num'] and (curr_word_num + 1) == lineObj['word_num']:
                temp_phrase.extend(lineObj['word'].split(' '))
                curr_word_num = lineObj['word_num']
            else:
                training.append({'language': 'aramaic','phrase': temp_phrase[:]})
                curr_line_num = lineObj['line_num']
                curr_word_num = lineObj['word_num']
                temp_phrase = lineObj['word'].split(' ')

    total_words = 0
    total_phrases = len(training)
    for p in training:
        total_words += len(p['phrase'])

    print 'NUM PHRASES: {} AVG WORDS PER PHRASE: {}'.format(total_phrases,total_words/total_phrases)

    return training

def make_mishnaic_training_context():
    training = []
    mishnah_indexes = [library.get_index(ind) for ind in library.get_indexes_in_category("Mishnah")]

    mishnah_indexes += [library.get_index(ind) for ind in library.get_indexes_in_category("Torah")]

    for ind in mishnah_indexes:
        mishna_segs = ind.all_section_refs()
        for seg in mishna_segs:
            first_sec_str = hebrew.strip_cantillation(seg.text('he').as_string(), strip_vowels=True)
            training += [{'language':'mishnaic', 'phrase': util.tokenize_words(p)} for p in first_sec_str.split(u'. ')]

    total_words = 0
    total_phrases = len(training)
    for p in training:
        total_words += len(p['phrase'])

    print 'NUM PHRASES: {} AVG WORDS PER PHRASE: {}'.format(total_phrases,total_words/total_phrases)
    return training

def merge_sets_context(a,m):
    full = []
    phrases =  a + m
    random.shuffle(phrases)

    num_phrases_per_section = 100
    i = 0
    while i < len(phrases):
        temp_phrases = phrases[i:i+num_phrases_per_section]
        full.append([{'l': p['language'][0], 'w': w} for p in temp_phrases for w in p['phrase']])
        i += num_phrases_per_section
    return full


dilate_lang()
