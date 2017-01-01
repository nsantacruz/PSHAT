# -*- coding: utf-8 -*-

import json, re, math, os
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
