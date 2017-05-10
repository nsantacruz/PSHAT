# -*- coding: utf-8 -*-

import json, re, math, os, heapq, codecs, decimal
from pprint import pprint
import local_settings
from sefaria.model import *


def get_ref_list(text_name,start_ref=None,end_ref=None):
    mesechta = library.get_index(text_name)
    if start_ref is None:
        start_ref = mesechta.all_section_refs()[0]
    if end_ref is None:
        end_ref = mesechta.all_section_refs()[-1]

    ref_list = []
    curr_ref = start_ref
    finished_yet = False
    while not finished_yet and not curr_ref is None:
        finished_yet = curr_ref == end_ref
        ref_list.append(curr_ref)
        curr_ref = curr_ref.next_section_ref()

    return ref_list

def saveUTFStr(obj,outFilename):
    objStr = json.dumps(obj, indent=4, ensure_ascii=False)
    with open(outFilename, "w") as f:
        f.write(objStr.encode('utf-8'))

def calClean(calIn):
    if type(calIn) == list:
        return [calClean(tempCalIn) for tempCalIn in calIn]
    else:
        return re.sub(r'[!?<>{}\+/\[\]\^\*\|#0-9\"\.]',"",calIn)


cal2hebDic = {
    ")": "א",
    "b": "ב",
    "g": "ג",
    "d": "ד",
    "h": "ה",
    "w": "ו",
    "z": "ז",
    "x": "ח",
    "T": "ט",
    "y": "י",
    "k": "כ",
    "K": "ך",
    "l": "ל",
    "m": "מ",
    "M": "ם",
    "n": "נ",
    "N": "ן",
    "s": "ס",
    "(": "ע",
    "p": "פ",
    "P": "ף",
    "c": "צ",
    "C": "ץ",
    "q": "ק",
    "r": "ר",
    "$": "שׁ",
    "&": "שׂ",
    "t": "ת",
    "@": " "  # TODO think about what @ means syntactically
}
def cal2heb(calIn,withshinsin=True):
    hebStr = ""


    no_shinsin_dic = {"$":"ש","&":"ש"}
    if type(calIn) == list:
       return [cal2heb(tempCalIn,withshinsin) for tempCalIn in calIn]
    else:
        for char in calIn:
            if char in cal2hebDic.keys():
                if char in no_shinsin_dic and not withshinsin:
                    hebStr += no_shinsin_dic[char]
                else:
                    hebStr += cal2hebDic[char]
            else:
                hebStr += char

    return unicode(hebStr,'utf-8')

def heb2cal(hebIn):
    calStr = ""
    inv_map = {unicode(v,'utf-8'): k for k, v in cal2hebDic.iteritems() if not k == '$' and not k == '&'}
    inv_map[u'ש'] = '$'
    if type(hebIn) == list:
        return [heb2cal(tempHebIn) for tempHebIn in hebIn]
    else:
        for char in hebIn:
            if char in inv_map.keys():
                calStr += inv_map[char]
            else:
                calStr += char

    return calStr

def daf2num(daf,side):
    return daf*2 + side - 2

def num2daf(num):
    daf = math.ceil(num/2.0)
    side = (num % 2)
    sideLetter = "a" if side == 1 else "b"
    return str(int(daf)) + sideLetter

def parseJBALine(line,shouldConvert,withshinsin=True):
    line = line.rstrip()
    lineArray = re.split(r'\t+', line)
    if len(lineArray) < 2: #this line doesn't have enough info
        return {}
    word = lineArray[0]
    synInfo = lineArray[1]
    prefixList = []
    prefix_POSList = []
    hasPrefix = False

    prefixRegStr = r'[a-zA-Z]+(#[0-9])?_? [a-zA-Z][0-9]{0,2}\+'
    prefixMultiPattern = re.compile(r'((' + prefixRegStr + ')+)')
    prefixSinglePattern = re.compile(prefixRegStr)
    prefixMultiMatch = prefixMultiPattern.match(synInfo)
    if prefixMultiMatch:
        hasPrefix = True
        prefixIter = prefixSinglePattern.finditer(prefixMultiMatch.group(1))
        for prefixInfo in prefixIter:
            prefixInfoArray = prefixInfo.group(0).split(" ")
            prefixList.append(prefixInfoArray[0])
            prefix_POSList.append(prefixInfoArray[1][:-1]) #skip +
        synInfo = synInfo[prefixMultiMatch.span()[1]:]
    synArray = synInfo.split(" ")
    head_word = synArray[0]
    POS = synArray[1]

    lineObj = {}
    if shouldConvert:
        lineObj["head_word"] = cal2heb(calClean(head_word),withshinsin=withshinsin)
        lineObj["POS"] = POS
        lineObj["word"] = cal2heb(calClean(word),withshinsin=withshinsin)
        if hasPrefix:
            lineObj["prefix"] = cal2heb(calClean(prefixList),withshinsin=withshinsin)
            lineObj["prefix_POS"] = prefix_POSList
    else:
        lineObj["head_word"] = calClean(head_word)
        lineObj["POS"] = POS
        lineObj["word"] = calClean(word)
        if hasPrefix:
            lineObj["prefix"] = calClean(prefixList)
            lineObj["prefix_POS"] = prefix_POSList

    return lineObj

def parseCalLine(line,shouldConvert,withshinsin=True):
    line = line.rstrip()
    lineArray = re.split(r'\t+', line)

    pos = lineArray[0]
    lineObj = {
        "book_num" : int(pos[0:5]),
        "ms" : pos[5:7],
        "pg_num" : int(pos[7:10]),
        "side" : int(pos[10:11]),
        "line_num" : int(pos[11:13])
    }

    word_num = lineArray[1]
    lineObj["word_num"] = int(word_num)

    synInfo = lineArray[2]
    prefixList = []
    prefix_POSList = []
    prefix_homograph_num_list = []
    hasPrefix = False

    calChars = r'a-zA-Z\)\(@\&\$'
    prefixRegStr = r'[' + calChars + r']+(#[0-9])?_? [' + calChars + r'][0-9]{0,2}\+'
    prefixMultiPattern = re.compile(r'((' + prefixRegStr + ')+)')
    prefixSinglePattern = re.compile(prefixRegStr)
    prefixMultiMatch = prefixMultiPattern.match(synInfo)
    if prefixMultiMatch:
        hasPrefix = True
        prefixIter = prefixSinglePattern.finditer(prefixMultiMatch.group(1))
        for prefixInfo in prefixIter:
            prefixInfoArray = prefixInfo.group(0).split(" ")
            prefixList.append(prefixInfoArray[0].split("#")[0])
            prefix_POSList.append(prefixInfoArray[1][:-1]) #skip +
            prefix_homograph_num_list.append(prefixInfo.group(1) if prefixInfo.group(1) else '')
        synInfo = synInfo[prefixMultiMatch.span()[1]:]

    synArray = synInfo.split(" ")

    head_word_split = synArray[0].split('#')
    head_word = cal2heb(calClean(head_word_split[0]),withshinsin=withshinsin) if shouldConvert else calClean(head_word_split[0])
    head_word_homograph = '#' + head_word_split[1] if len(head_word_split) > 1 else ''

    POS = synArray[1]
    word = cal2heb(calClean(synArray[2]),withshinsin=withshinsin) if shouldConvert else calClean(synArray[2])

    prefix = cal2heb(calClean(prefixList),withshinsin=withshinsin) if shouldConvert else calClean(prefixList)


    lineObj["head_word"] = head_word
    lineObj["homograph"] = head_word_homograph
    lineObj["POS"] = POS
    lineObj["word"] = word
    if hasPrefix:
        lineObj["prefix"] = prefix
        lineObj["prefix_POS"] = prefix_POSList
        lineObj["prefix_homograph"] = prefix_homograph_num_list

    return lineObj


def writeCalLine(lo):
    """
    I could probably write this in one line...but that's probably a bad idea
    :param lo: line_object coming from parseCalLine()
    :return:
    """
    if "prefix" in lo:
        prefix = "+".join(["{}{} {}".format(prefix,prefixHomo,prefixPOS) for prefix,prefixPOS,prefixHomo in zip(lo["prefix"],lo["prefix_POS"],lo["prefix_homograph"])])
        full_pos = "{}+{}{} {} {}".format(prefix, lo["head_word"], lo["homograph"], lo["POS"], lo["word"])
    else:
        full_pos = "{}{} {} {}".format(lo["head_word"], lo["homograph"], lo["POS"], lo["word"])


    return "{book_num:05d}{ms}{pg_num:03d}{side}{line_num:02d}\t{word_num}\t{full_pos}".format(book_num=lo["book_num"],ms=lo["ms"],pg_num=lo["pg_num"],side=lo["side"],line_num=lo["line_num"],word_num=lo["word_num"],full_pos=full_pos)

def calLine2hebLine(calLine):
    lineObj = parseCalLine(calLine,True)
    return str(lineObj["book_num"]) + \
        str(lineObj["ms"]) + str(lineObj["pg_num"]) + str(lineObj["side"]) + \
        str(lineObj["line_num"]) + "\t" + str(lineObj["word_num"]) + "\t" + \
        lineObj["head_word"] + " " + lineObj["POS"] + " " + lineObj["word"]

def caldb2hebdb(calFilename,hebFilename):
    with open(calFilename,'r') as cal:
        with open(hebFilename,'w') as heb:
            for calLine in cal:
                hebLine = calLine2hebLine(calLine)
                heb.write(hebLine + "\n")

def make_folder_if_need_be(path):
    if not os.path.exists(path):
        os.makedirs(path)

def argmax(iterable, n=1):
    if n==1:
        return max(enumerate(iterable), key=lambda x: x[1])[0]
    else:
        return heapq.nlargest(n, xrange(len(iterable)), iterable.__getitem__)


def tokenize_words(str):
    str = str.replace(u"־", " ")
    str = re.sub(r"</?.+>", "", str)  # get rid of html tags
    str = re.sub(r"\([^\(\)]+\)", "", str)  # get rid of refs
    # str = str.replace("'", '"')
    word_list = filter(bool, re.split(r"[\s\:\-\,\.\;\(\)\[\]\{\}]", str))
    return word_list

def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += float(decimal.Decimal(jump))


def stats_on_lang_mixing():
    from os.path import join
    import random

    random.seed(2823274491)
    dir = 'data/5_pos_tagged/json/'
    min_seg_length = 5  # min length of segment. if shorter, append the next segment with it

    all_json_files = []
    # collect all the individual filenames
    for dirpath, dirnames, filenames in os.walk(dir):
        all_json_files.extend([join(dirpath, filename) for filename in filenames if
                               filename.endswith('.json')])

    total_words = 0
    total_daf = 0
    total_segs = 0

    random.shuffle(all_json_files)
    percent_training = 0.2
    split_index = int(round(len(all_json_files) * percent_training))
    train_data = all_json_files[split_index:]
    val_data = all_json_files[:split_index]
    # iterate through all the files, and load them in
    for i in range(10):
        print val_data[i]
    segments = []
    for file in val_data:

        with codecs.open(file, 'r', encoding='utf8') as f:
            all_text = f.read()

        # parse
        daf_data = json.loads(all_text)

        all_words = []
        for word in daf_data['words']:
            word_s = word['word']
            # class will be 1 if talmud, 0 if unknown
            word_class = word['gold_class']
            word_lang = word['lang']
            word_pos = word['gold_pos']
            got_it_right = 1 if word['gold_pos'] == word['predicted'] else 0
            total_words += 1

            all_words.append((word_s, word_class, word_pos, word_lang, got_it_right))

        total_daf += 1
        # yield it
        split_file = file.split('/')
        mesechta_name = split_file[split_file.index('json') + 1]
        daf_num = split_file[-1].split('.json')[0].split('_')[1]
        daf = {"words": all_words, "file": '{}_{}'.format(mesechta_name, daf_num)}

        # break up daf into segments
        daf_chunk = Ref("{} {}".format(mesechta_name, daf_num)).text("he")
        ind_list, ref_list, total_len = daf_chunk.text_index_map(tokenize_words)

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


    #now you have the segments


    # calculate percent and success for diff buckets of Aramaic

    percents = [0 for _ in range(20)]
    success = [0 for _ in range(20)]

    num_cal = [0 for _ in range(20)]
    for seg in segments:
        num_aramaic = 0.0
        success_seg = 0.0
        num_cal_seg = 0.0

        for word in seg['words']:
            num_aramaic += word[3]
            if word[1] == 1:
                success_seg += word[4]
                num_cal_seg += 1

        percent_aramaic = num_aramaic / len(seg['words'])
        bucketed = int(round(percent_aramaic * 20))
        if bucketed >= len(percents):
            bucketed = len(percents) - 1
        percents[bucketed] += 1
        success[bucketed] += success_seg
        num_cal[bucketed] += num_cal_seg

    #print "{}/{} {}".format(num_aramaic, total_num, 1.0*num_aramaic/total_num)  # 165511/519630 0.318517021727
    percents = [1.0 * p / len(segments) for p in percents]
    success = [1.0 * s / c if c != 0 else 0.0 for s,c in zip(success,num_cal)]
    print pprint(percents)
    print pprint(success)


def make_pos_hashtable(data):
    from collections import defaultdict
    pos_hashtable = defaultdict(lambda: False)
    head_word_hashtable = defaultdict(lambda: False)
    for daf in data:
        for word_s, word_class, word_pos, head_word, word_lang, got_it_right in daf["words"]:
            pos_hashtable[word_s] = True
            head_word_hashtable[head_word] = True

    return pos_hashtable, head_word_hashtable


def make_best_pos_hashtable(data):
    from collections import defaultdict
    import numpy as np
    possible_pos = defaultdict(int)
    temp_pos_hashtable = {}
    temp_hw_pos_hashtable = {}
    for daf in data:
        for w, w_class, w_pos, hw, word_lang, got_it_right in daf["words"]:
            if w_class:
                if not w in temp_pos_hashtable:
                    temp_pos_hashtable[w] = defaultdict(int)
                temp_pos_hashtable[w][w_pos] += 1
                possible_pos[w_pos] += 1

                if not hw in temp_hw_pos_hashtable:
                    temp_hw_pos_hashtable[hw] = defaultdict(int)
                temp_hw_pos_hashtable[hw][w_pos] += 1


    real_pos_hashtable = {}
    real_hw_pos_hashtable = {}
    ambig_list = np.zeros(len(temp_pos_hashtable))
    counter = defaultdict(int)
    i = 0
    for w,pos_dict in temp_pos_hashtable.items():
        best_pos_list = sorted(pos_dict.items(),key=lambda x: x[1])
        ambig_list[i] = len(best_pos_list)
        real_pos_hashtable[w] = best_pos_list[0][0]
        counter[len(best_pos_list)] += 1
        i += 1

    for hw, hw_pos_dict in temp_hw_pos_hashtable.items():
        best_hw_pos_list = sorted(hw_pos_dict.items(), key=lambda x: x[1])
        real_hw_pos_hashtable[hw] = best_hw_pos_list[0][0]

    #print counter
    #print 'Max {} Avg {} Var {}'.format(np.max(ambig_list), np.average(ambig_list), np.var(ambig_list))

    best_pos = sorted(possible_pos.items(), key=lambda x: x[1])[-1][0]

    return real_pos_hashtable, real_hw_pos_hashtable, best_pos
def unknown_word_stats():
    from os.path import join
    import random

    dir = 'data/5_pos_tagged/json/'
    old_dir = 'data/2_matched_sefaria/json/'

    all_json_files = []
    # collect all the individual filenames
    for dirpath, dirnames, filenames in os.walk(dir):
        all_json_files.extend([join(dirpath, filename) for filename in filenames if
                               filename.endswith('.json')])

    all_old_json_files = []
    # collect all the individual filenames
    for dirpath, dirnames, filenames in os.walk(old_dir):
        all_old_json_files.extend([join(dirpath, filename) for filename in filenames if
                               filename.endswith('.json')])

    total_words = 0
    total_daf = 0
    all_data = []
    # iterate through all the files, and load them in
    for file, old_file in zip(all_json_files, all_old_json_files):
        #print file, old_file
        with codecs.open(file, 'r', encoding='utf8') as f:
            all_text = f.read()

        with codecs.open(old_file, 'r', encoding='utf8') as f:
            all_old_text = f.read()

        # parse
        daf_data = json.loads(all_text)
        old_daf_data = json.loads(all_old_text)

        all_words = []
        for word, old_word in zip(daf_data['words'], old_daf_data['words']):


            word_s = word['word']
            #print word_s, old_word['word']
            assert word_s == old_word['word']

            # class will be 1 if talmud, 0 if unknown
            word_class = word['gold_class']
            word_lang = word['lang']
            word_pos = word['gold_pos']
            head_word = old_word['head_word'] if 'head_word' in old_word else ''
            got_it_right = 1 if word['gold_pos'] == word['predicted'] else 0
            total_words += 1

            all_words.append((word_s, word_class, word_pos, head_word, word_lang, got_it_right))

        total_daf += 1
        all_data += [{"words": all_words, "file": file}]

    # see how it dealt with unknown words

    random.seed(2823274491)
    random.shuffle(all_data)
    percent_training = 0.2
    split_index = int(round(len(all_data) * percent_training))
    train_data = all_data[split_index:]
    val_data = all_data[:split_index]
    for i in range(10):
        print val_data[i]

    pos_hashtable, head_word_table, best_pos = make_best_pos_hashtable(train_data)

    known_count = [0.0, 0.0]
    known_head_word_unknown_word_count = [0.0, 0.0]
    unknown_word_count = [0.0, 0.0]

    bl_known_count = [0.0, 0.0]
    bl_known_head_word_unknown_word_count = [0.0, 0.0]
    bl_unknown_word_count = [0.0, 0.0]

    for daf in val_data:
        for word_s, word_class, word_pos, head_word, word_lang, got_it_right in daf["words"]:
            if word_class == 1:
                if word_s in pos_hashtable:
                    known_count[got_it_right] += 1
                    bl_known_count[int(word_pos == pos_hashtable[word_s])] += 1
                elif head_word in head_word_table:
                    known_head_word_unknown_word_count[got_it_right] += 1
                    bl_known_head_word_unknown_word_count[int(word_pos == head_word_table[head_word])] += 1
                else:
                    unknown_word_count[got_it_right] += 1
                    bl_unknown_word_count[int(word_pos == best_pos)] += 1


    print "KNOWN {}/{} {}".format(known_count[1], sum(known_count), known_count[1] / sum(known_count))
    print "KNOWN HW {}/{} {}".format(known_head_word_unknown_word_count[1], sum(known_head_word_unknown_word_count), known_head_word_unknown_word_count[1] / sum(known_head_word_unknown_word_count))
    print "UNK   {}/{} {}".format(unknown_word_count[1], sum(unknown_word_count), unknown_word_count[1] / sum(unknown_word_count))


    print "KNOWN {}/{} {}".format(bl_known_count[1], sum(bl_known_count), bl_known_count[1] / sum(bl_known_count))
    print "KNOWN HW {}/{} {}".format(bl_known_head_word_unknown_word_count[1], sum(bl_known_head_word_unknown_word_count),
                                     bl_known_head_word_unknown_word_count[1] / sum(bl_known_head_word_unknown_word_count))
    print "UNK   {}/{} {}".format(bl_unknown_word_count[1], sum(bl_unknown_word_count),
                                  bl_unknown_word_count[1] / sum(bl_unknown_word_count))
#stats_on_lang_mixing()
unknown_word_stats()