# -*- coding: utf-8 -*-
import math
import re, os
import numpy as np
import json
import codecs
import csv
import util
import local_settings
from sefaria.model import *

calDBRoot = "cal DB"
calDBFile = calDBRoot + "/bavliwords.txt"
sefariaRoot = "sefaria talmud"



def saveHeadwordHashtable(calFilename,outFilename):
    hash = {}
    with open(calFilename,"r") as cal:
        for calLine in cal:
            lineObj = util.parseCalLine(calLine,True)
            try:
                tempSet = hash[lineObj["head_word"]]
                tempSet.add(lineObj["word"])
            except KeyError:
                tempSet = set()
                tempSet.add(lineObj["word"])
                hash[lineObj["head_word"]] = tempSet
    with open(outFilename,"w") as out:
        keys = hash.keys()
        keys.sort()
        for key in keys:
            listStr = ""
            isFirst = True
            for el in hash[key]:
                listStr = listStr + el if isFirst else listStr + ", " + el
                isFirst = False
            out.write("%s - %s\n" % (key,listStr))

def savePOSHashtable(calFilename,outFilename):
    obj = {}
    with open(calFilename,"r") as cal:
        for calLine in cal:
            lineObj = util.parseCalLine(calLine,True)
            try:
                tempSet = set(obj[lineObj["word"]])
                tempSet.add(lineObj["POS"])
                obj[lineObj["word"]] = list(tempSet)
            except KeyError:
                obj[lineObj["word"]] = [lineObj["POS"]]

    posCountList = np.empty(shape=[len(obj.keys())])
    with codecs.open(outFilename + ".txt","w",encoding="utf-8") as out:
        keys = obj.keys()
        keys.sort()
        for i,key in enumerate(keys):
            listStr = ""
            isFirst = True
            posCountList[i] = (len(obj[key]))
            for el in obj[key]:
                listStr = listStr + el if isFirst else listStr + ", " + el
                isFirst = False
            out.write("%s - %s\n" % (key,listStr))

    util.saveUTFStr(obj,outFilename + ".json")
    print "AVG POS: %s" % np.mean(posCountList)
    print "VAR POS: %s" % np.var(posCountList)

def saveJBAForms(outFilename):
    obj = {}
    with open("jbaforms.txt", 'rb') as jba:
        for line in jba:
            lineObj = util.parseJBALine(line,True,False)
            if lineObj == {}:
                continue
            try:
                obj[lineObj["word"]].append(lineObj)
            except KeyError:
                obj[lineObj["word"]] = [lineObj]

    for word in obj:
        pos_dic = {}
        head_dic = {}
        for lineObj in reversed(obj[word]):
            if lineObj["POS"] in pos_dic and lineObj["head_word"] in head_dic:
                obj[word].remove(lineObj)
            else:
                pos_dic[lineObj["POS"]] = True
                head_dic[lineObj["head_word"]] = True


    util.saveUTFStr(obj, outFilename + ".json")



def stupidTagger(outFilename):
    jba = json.load(open(calDBRoot + "/POSHashtable.json","r"),encoding="utf8")
    sef = json.load(open(sefariaRoot + "/Berakhot.json","r"),encoding="utf8")
    daf = sef["text"][3]
    with codecs.open(outFilename,"w","utf-8") as out:
        for line in daf:
            posLine = ""
            lineArray = line.rstrip().split(" ")
            for word in lineArray:
                try:
                    posList = list(jba[word])
                    if len(posList) >= 1:
                        posLine += "%s(%s) " % (word,posList)
                except KeyError:
                    posLine += "%s(UKN) " % word
            out.write(posLine+"\n")

def fix_wrong_pos_in_dataset():

    newCalDb = [l for l in open("caldb.txt","r").read().split('\n') if len(l) > 0]
    wordFixerDict = {}
    numfixed = 0
    with open("double_pos_checker.txt","r") as f:
        for line in f:
            if "XXX" in line:
                word = re.findall("\^.+\^",line)[0].split('^')[1]
                headPOSs = re.findall("\[(.+)\]",line)[0].replace("'","").split(", ")
                headWords = line.split(' ')[1:2][0].split('*-*')
                newHeadPOSs = re.findall("XXX\s(.+)",line)[0].split(',')
                if word in wordFixerDict:
                    print 'oh no!'
                wordFixerDict[word] = {"headPOSs":headPOSs,"headWords":headWords,"newHeadPOSs":newHeadPOSs}
    for word in wordFixerDict:
        badPOSs = [pos for pos in wordFixerDict[word]["headPOSs"] if pos not in wordFixerDict[word]["newHeadPOSs"]]
        canFix = len(wordFixerDict[word]["newHeadPOSs"]) == 1
        if canFix:
            for iline,line in enumerate(newCalDb):
                lineObj = util.parseCalLine(line,False)
                if lineObj["word"] == word and lineObj["POS"] in badPOSs:
                    if not 'prefix' in lineObj and lineObj["head_word"][-1] == '_':
                        lineObj['prefix'] = [lineObj['head_word']]
                        lineObj['prefix_POS'] = [lineObj['POS']]
                        lineObj['prefix_homograph'] = [lineObj['homograph']]
                    lineObj["POS"] = wordFixerDict[word]["newHeadPOSs"][0]
                    lineObj["head_word"] = wordFixerDict[word]["headWords"][wordFixerDict[word]["headPOSs"].index(wordFixerDict[word]["newHeadPOSs"][0])]
                    lineObj["homograph"] = ''

                    new_line = util.writeCalLine(lineObj)
                    numfixed += 1
                    newCalDb[iline] = new_line
    ncaldb = open("noahcaldb.txt","w")
    for new_line in newCalDb:
        ncaldb.write(new_line+'\n')
    ncaldb.close()
    print numfixed


def fixPNandGN():
    ncaldb = open("noahcaldbfull.txt", 'w')
    numfixed = 0
    with open("bavliwordsfull.txt", 'r') as f:
        for line in f:
            line = line[:-1]
            lineObj = util.parseCalLine(line, False)
            if ('-' in lineObj["word"] or '=' in lineObj["word"]) and len(lineObj['head_word']) > 0 and lineObj["head_word"][-1] == '_':
                prefix = lineObj["head_word"]
                prefixPOS = lineObj["POS"]
                prefixHomo = lineObj["homograph"]
                lineObj["prefix"] = [prefix]
                lineObj["prefix_POS"] = [prefixPOS]
                lineObj["prefix_homograph"] = [prefixHomo]

                split_index = lineObj["word"].find('-')
                split_index = lineObj["word"].find('=') if split_index == -1 else split_index

                lineObj["head_word"] = lineObj["word"][split_index + 1:]
                lineObj["POS"] = "PN" if '-' in lineObj["word"] else "GN"
                lineObj["word"] = lineObj["word"][:split_index] + lineObj["word"][split_index + 1:]
                lineObj["homograph"] = ''

                new_line = util.writeCalLine(lineObj)
                print new_line
                ncaldb.write(new_line + '\n')
                numfixed +=1
            else:
                ncaldb.write(util.writeCalLine(lineObj) + '\n')

    ncaldb.close()
    print numfixed


def split_training_set_into_mesechtot():
    mesechta_map = {71001:"Berakhot",71002:"Shabbat",71003:"Eruvin",71004:"Pesachim",71020:"Bava Kamma",71021:"Bava Metzia",71022:"Bava Batra"}
    curr_book = -1
    curr_book_file = None
    with open("data/1_cal_input/caldbfull.txt","r") as ncal:
        for line in ncal:
            lo = util.parseCalLine(line,False)
            if lo["book_num"] != curr_book:
                curr_book = lo["book_num"]
                if curr_book_file:
                   curr_book_file.close()
                curr_book_file = open("data/1_cal_input/caldb_{}.txt".format(mesechta_map[curr_book]),'w')
            if lo['ms'] == '01':
                curr_book_file.write(line)
        if curr_book_file:
            curr_book_file.close()

def print_tagged_corpus_to_html_table(text_name, ref_list, num_daf_per_doc):
    cal_dh_root = "data/2_matched_sefaria"

    iref = 0
    while iref < len(ref_list):
        str = u"<html><head><style>h1{text-align:center;background:grey}td{text-align:center}table{margin-top:20px;margin-bottom:20px;margin-right:auto;margin-left:auto;width:1200px}.missed{color:white;background:red}.b{color:green}.m{color:blue}.sef{color:black}.cal{color:grey}.good-cal{color:red}.good-jba{background:#eee;color:red}.POS{color:orange}</style><meta charset='utf-8'></head><body>"

        start_daf = ""
        end_daf = ""
        for idaf in xrange(num_daf_per_doc):
            if iref >= len(ref_list): break
            ref = ref_list[iref]
            daf = ref.__str__().replace("{} ".format(text_name), "").encode('utf8')
            str += u"<h1>DAF {}</h1>".format(daf)
            str += u"<table>"
            if idaf == 0: start_daf = daf
            if idaf == num_daf_per_doc - 1: end_daf = daf

            try:
                util.make_folder_if_need_be('{}/json/{}'.format(cal_dh_root, text_name))
                test_set = json.load(codecs.open(
                    "{}/json/{}/{}.json".format(cal_dh_root, text_name, daf),
                    "r", encoding="utf-8"))
            except IOError:
                continue  # this daf apparently didn't exist in cal dataset but does in sefaria
            word_list = test_set["words"]
            missed_word_list = test_set["missed_words"]
            missed_dic = {wo["index"]: wo["word"] for wo in missed_word_list}

            sef_count = 0
            cal_count = 0
            while sef_count < len(word_list):
                row_obj = word_list[sef_count:sef_count + 10]
                row_sef = u"<tr class='sef'><td>{}</td>".format(
                    u"</td><td>".join([wo["word"] for wo in reversed(row_obj)]))
                row_sef += u"<td>({}-{})</td></tr>".format(sef_count, sef_count + len(row_obj) - 1)

                row_cal = u"<tr class='cal'>"
                start_cal_count = cal_count
                for wo in reversed(row_obj):
                    while cal_count in missed_dic:
                        cal_count += 1
                    if "cal_word" in wo:
                        cal_count += 1
                        row_cal += u"<td class='good-cal'>{} <span class='POS'>({})</span></td>".format(
                            wo["cal_word"], wo["POS"])
                    elif "jba_word" in wo:
                        row_cal += u"<td class='good-jba'>{} <span class='POS'>({})</span><br>{}</td>".format(
                            wo["jba_word"], wo["POS"], wo["head_word"])
                    else:
                        row_cal += u"<td class='{}'>{}</td>".format(wo["class"][0], wo["class"][0:3].upper())
                row_cal += u"<td>({}-{})</td>".format(start_cal_count, cal_count - 1)
                row_cal += u"</tr>"

                str += row_sef
                str += row_cal
                sef_count += 10
            str += u"</table>"

            str += u"<table>"
            count = 0
            while count < len(missed_word_list):
                row_obj = missed_word_list[count:count + 10]
                word_str = [u"{}:{}".format(wo["word"], wo["index"]) for wo in reversed(row_obj)]
                row_missed = u"<tr class='missed'><td>{}</td></tr>".format(u"</td><td>".join(word_str))
                str += row_missed
                count += 10
            str += u"</table>"
            iref += 1
        str += u"</body></html>"
        util.make_folder_if_need_be('{}/html/{}'.format(cal_dh_root, text_name))
        fp = codecs.open("{}/html/{}/{}-{}.html".format(cal_dh_root, text_name, start_daf, end_daf), 'w',
                         encoding='utf-8')
        fp.write(str)
        fp.close()
#print cal2heb("(ny")
#print daf2num(21,1)
#print num2daf(3)

#caldb2hebdb("cal DB/bavliwords.txt","cal DB/bavliwordsHE.txt")
#saveHeadwordHashtable("cal DB/bavliwords.txt","cal DB/headwordHashtable.txt")
#savePOSHashtable("cal DB/bavliwords.txt","cal DB/POSHashtable")
#saveJBAForms("JBAHashtable")
#stupidTagger("output/stupidBerakhot.txt")
#fix_wrong_pos_in_dataset()
#fixPNandGN()

#split_training_set_into_mesechtot()
for mesechta in local_settings.MESECHTA_NAMES:
    ref_list = util.get_ref_list(mesechta,None,None)
    print_tagged_corpus_to_html_table(mesechta, ref_list, 10)


