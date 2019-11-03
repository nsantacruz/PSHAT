# -*- coding: utf-8 -*-
import sys
import random,json,codecs
from collections import OrderedDict

import util
import dynet as dy
import numpy as np
import argparse,re,codecs
from os import listdir
from os.path import isfile, join
import timeit
import time

def _template_func(setup, func):
    """Create a timer function. Used if the "statement" is a callable."""
    def inner(_it, _timer, _func=func):
        setup()
        _t0 = _timer()
        for _i in _it:
            retval = _func()
        _t1 = _timer()
        return _t1 - _t0, retval
    return inner

timeit._template_func = _template_func

def timereturn(fun, iters, setup):
    t = timeit.Timer(fun)
    times = []
    results = []
    for i in xrange(0, iters):
        setup()
        time1, result = t.timeit(number=1)
        times.append(time1)
        results.append(result)

    return (times, results)

# set the seed
random.seed(2823274491)
model_root = 'data/3_lang_tagged/model'
filename_to_load  = ''
START_EPOCH = 0

# How many epochs should we wait if the validation error doesn't decrease
EARLY_STOP_PATIENCE_N_EPOCHS = 3


# argument parse
parser = argparse.ArgumentParser()
parser.add_argument('-hiddim', '-hiddendim', help='Size of the RNN hidden layer, default 100', default=40,
                    required=False)
parser.add_argument('-embeddim', '-embeddingdim', help='Size of the embeddings, default 50', default=20, required=False)
parser.add_argument('-layers', '-mlplayers', help='Number of MLP layers, can only be 2 or 3', default=2, required=False)
parser.add_argument('-bilstmlayers', '-lstmlayers', help='Number of BILSTM layers, default 2', default=2,
                    required=False)
parser.add_argument('-model', '-modeltoload', help='Filename of model to load', default='', required=False)
args = vars(parser.parse_known_args()[0])

# get the params
HIDDEN_DIM = int(args['hiddim'])
EMBED_DIM = int(args['embeddim'])
BILSTM_LAYERS = int(args['bilstmlayers'])
fDo_3_Layers = int(args['layers']) == 3
sLAYERS = '3' if fDo_3_Layers else '2'
Filename_to_log = '{}/postagger_log_embdim{}_hiddim{}_lyr{}.txt'.format(model_root,EMBED_DIM,HIDDEN_DIM,sLAYERS)


def log_message(message):
    print message
    with codecs.open(Filename_to_log, "a", encoding="utf8") as myfile:
        myfile.write("\n" + message)


if args['model']:
    filename_to_load = args['model']
    START_EPOCH = int(re.search("_e(\d+)", filename_to_load).group(1)) + 1

log_message('EMBED_DIM: ' + str(EMBED_DIM))
log_message('HIDDEN_DIM: ' + str(HIDDEN_DIM))
log_message('BILSTM_LAYERS: ' + str(BILSTM_LAYERS))
log_message('MLP Layers: ' + sLAYERS)
if filename_to_load:
    log_message('Loading model: ' + filename_to_load)
    log_message('Starting epoch: ' + str(START_EPOCH))


def read_data(dir=''):
    if not dir:
        dir = '{}/lstm_training.json'.format(model_root)
    training_set = json.load(codecs.open(dir, "rb", encoding="utf-8"))

    tags = ['aramaic','mishnaic','ambiguous']
    training_set = [{'word':w['word'],'tag':tags.index(w['tag'])} for w in training_set]
    return training_set


# Classes:
# 1] Vocabulary class (the dictionary for char-to-int)
# 2] WordEncoder (actually, it'll be a char encoder)
# 3] Simple character BiLSTM
# 4] MLP
# 5] ConfusionMatrix
class Vocabulary(object):
    def __init__(self):
        self.all_items = []
        self.c2i = {}

    def add_text(self, paragraph):
        self.all_items.extend(paragraph)

    def finalize(self, fAddBOS=True):
        self.vocab = sorted(list(set(self.all_items)))
        c2i_start = 1 if fAddBOS else 0
        self.c2i = {c: i for i, c in enumerate(self.vocab, c2i_start)}
        self.i2c = self.vocab
        if fAddBOS:
            self.c2i['*BOS*'] = 0
            self.i2c = ['*BOS*'] + self.vocab
        self.all_items = None

    # debug
    def get_c2i(self):
        return self.c2i

    def size(self):
        return len(self.i2c)

    def __getitem__(self, c):
        return self.c2i.get(c, 0)

    def getItem(self, i):
        return self.i2c[i]


class WordEncoder(object):
    def __init__(self, name, dim, model, vocab):
        self.vocab = vocab
        self.enc = model.add_lookup_parameters((vocab.size(), dim))

    def __call__(self, char, DIRECT_LOOKUP=False):
        char_i = char if DIRECT_LOOKUP else self.vocab[char]
        return dy.lookup(self.enc, char_i)


class MLP:
    def __init__(self, model, name, in_dim, hidden_dim, out_dim):
        self.mw = model.add_parameters((hidden_dim, in_dim))
        self.mb = model.add_parameters((hidden_dim))
        if not fDo_3_Layers:
            self.mw2 = model.add_parameters((out_dim, hidden_dim))
            self.mb2 = model.add_parameters((out_dim))
        if fDo_3_Layers:
            self.mw2 = model.add_parameters((hidden_dim, hidden_dim))
            self.mb2 = model.add_parameters((hidden_dim))
            self.mw3 = model.add_parameters((out_dim, hidden_dim))
            self.mb3 = model.add_parameters((out_dim))

    def __call__(self, x):
        mlp_output = self.mw2 * (dy.tanh(self.mw * x + self.mb)) + self.mb2
        if fDo_3_Layers:
            mlp_output = self.mw3 * (dy.tanh(mlp_output)) + self.mb3
        return dy.softmax(mlp_output)


class BILSTMTransducer:
    def __init__(self, LSTM_LAYERS, IN_DIM, OUT_DIM, model):
        self.lstmF = dy.LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)
        self.lstmB = dy.LSTMBuilder(LSTM_LAYERS, IN_DIM, (int)(OUT_DIM / 2), model)

    def __call__(self, seq):
        """
        seq is a list of vectors (either character embeddings or bilstm outputs)
        """
        fw = self.lstmF.initial_state()
        bw = self.lstmB.initial_state()
        outf = fw.transduce(seq)
        outb = list(reversed(bw.transduce(reversed(seq))))
        return [dy.concatenate([f, b]) for f, b in zip(outf, outb)]


class ConfusionMatrix:
    def __init__(self, size, vocab):
        self.matrix = np.zeros((size, size))
        self.size = size
        self.vocab = vocab

    def __call__(self, x, y):
        self.matrix[x, y] += 1

    def to_html(self):
        fp_matrix = np.sum(self.matrix, 1)
        fn_matrix = np.sum(self.matrix, 0)

        html = """
                    <html>
                        <head>
                            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
                            <script src="confused.js"></script>
                            <style>.good{background-color:green;color:white}.bad{background-color:red;color:white}table{table-layout:fixed}td{text-align:center;padding:10px;border:solid 1px black}</style>
                        </head>
                        <body><h2>A Confusing Matrix</h2><table>"""
        first_row = "<tr><td></td>"
        for i in range(self.size):
            first_row += "<td data-col-head={}>{}</td>".format(i, self.vocab[i])
        first_row += "<td>False Positives</td></tr>"
        html += first_row
        for i in range(self.size):
            html += "<tr><td data-row-head={}>{}</td>".format(i, self.vocab[i])
            for j in range(self.size):
                classy = "good" if i == j else "bad"
                opacity = self.matrix[i, j] / (np.mean(self.matrix[self.matrix > 0]))
                if opacity < 0.2: opacity = 0.2
                if opacity > 1.0: opacity = 1.0
                html += "<td data-i={} data-j={} class=\"{}\" style=\"opacity:{}\">{}</td>".format(i, j, classy,
                                                                                                   opacity,
                                                                                                   self.matrix[i, j])

            html += "<td>{}</td></tr>".format(round(100.0 * (fp_matrix[i] - self.matrix[i, i]) / fp_matrix[i], 2))
        # add confusion table for each class
        stats = {"precision": self.precision, "recall": self.recall, "F1": self.f1}

        html += "<tr><td>False Negatives</td>"
        for i in range(self.size):
            html += "<td>{}</td>".format(round(100.0 * (fn_matrix[i] - self.matrix[i, i]) / fn_matrix[i], 2))
        html += "</tr>"

        for k, v in stats.items():
            html += "<tr><td>{}</td>".format(k)
            for j in range(self.size):
                tp = self.matrix[j, j]
                fp = fp_matrix[j] - tp
                fn = fn_matrix[j] - tp
                html += "<td>{}</td>".format(round(100 * v(tp, fp, fn), 2))
            html += "</tr>"
        html += "</table><h2>Table of Confusion</h2>"
        total_tp = sum([self.matrix[i, i] for i in range(self.size)])
        total_fp = np.sum(fp_matrix) - total_tp
        total_fn = np.sum(fn_matrix) - total_tp
        html += "<h3>Precision: {}</h3>".format(round(100 * self.precision(total_tp, total_fp, total_fn), 2))
        html += "<h3>Recall: {}</h3>".format(round(100 * self.recall(total_tp, total_fp, total_fn), 2))
        html += "<h3>F1: {}</h3>".format(round(100 * self.f1(total_tp, total_fp, total_fn), 2))

        html += "</body></html>"
        return html

    def f1(self, tp, fp, fn):
        return 2.0 * tp / (2.0 * tp + fp + fn) if tp + fp + fn != 0 else 0.0

    def recall(self, tp, fp, fn):
        return 1.0 * tp / (tp + fn) if tp + fn != 0 else 0.0

    def precision(self, tp, fp, fn):
        return 1.0 * tp / (tp + fp) if tp + fn != 0 else 0.0

    def clear(self):
        self.matrix = np.zeros((self.size, self.size))


# When fValidation is true and fRunning is false
# return (1 for true prediction; 0 for false) and a dict with word, predicted_lang, gold_lang and
# confidence
def CalculateLossForWord(word_obj, fValidation=False, fRunning=False):
    dy.renew_cg()

    if not fRunning: gold_lang = word_obj['tag']
    # add a bos before and after
    seq = ['*BOS*'] + list(word_obj['word']) + ['*BOS*']

    # get all the char encodings for the daf
    char_embeds = [let_enc(let) for let in seq]

    # run it through the bilstm
    char_bilstm_outputs = bilstm(char_embeds)
    bilistm_output = dy.concatenate([char_bilstm_outputs[0],char_bilstm_outputs[-1]])

    mlp_input = bilistm_output
    mlp_out = lang_mlp(mlp_input)
    try:
        temp_lang_array = mlp_out.npvalue()
        possible_lang_array = np.zeros(temp_lang_array.shape)
        possible_lang_indices = list(lang_hashtable[word_obj['word']])
        possible_lang_array[possible_lang_indices] = temp_lang_array[possible_lang_indices]
    except KeyError:
        possible_lang_array = mlp_out.npvalue()

    predicted_lang = lang_tags[np.argmax(possible_lang_array)]
    confidence = (mlp_out.npvalue()[:2] / np.sum(mlp_out.npvalue()[:2])).tolist() #skip ambiguous
    # if we aren't doing validation, calculate the loss
    if not fValidation and not fRunning:
        loss = -dy.log(dy.pick(mlp_out, gold_lang))
    # otherwise, set the answer to be the argmax
    elif not fRunning and fValidation:
        loss = None
        lang_conf_matrix(np.argmax(mlp_out.npvalue()), gold_lang)
    else:
        return predicted_lang,confidence

    pos_prec = 1 if predicted_lang == lang_tags[gold_lang] else 0

    tagged_word = { 'word': word_obj['word'], 'tag': predicted_lang, 'confidence':confidence, 'gold_tag':lang_tags[gold_lang]}

    if fValidation:
        return pos_prec, tagged_word

    return loss, pos_prec


def run_network_on_validation(epoch_num):
    return run_network_on_data(epoch_num, val_data, "Validation", True)



def run_network_on_test():
    return run_network_on_data(-1, test_data, "Test", False)


# Return accuracy (TP + TN / N) of model on validation set `test_data`
def run_network_on_data(epoch_num, data, data_name, shouldPersist):
    data_lang_prec = 0.0
    data_lang_items = 0
    # iterate
    num_words_to_save = 1000
    words_to_save = []


    for idaf, word in enumerate(data):
        lang_prec, tagged_word = CalculateLossForWord(word, fValidation=True)
        # increment and continue
        data_lang_prec += lang_prec
        data_lang_items += 1
        if epoch_num >= 0 and idaf % round(1.0 * len(data) / num_words_to_save) == 0:
            words_to_save.append(tagged_word)

    # divide
    data_lang_prec = data_lang_prec / data_lang_items * 100 if data_lang_items > 0 else 0.0
    # print the results
    log_message('{} accuracy: {}%'.format(data_name, data_lang_prec))

    if shouldPersist:
        objStr = json.dumps(words_to_save, indent=4, ensure_ascii=False)
        util.make_folder_if_need_be('{}/epoch_{}'.format(model_root,epoch_num))
        with open("{}/epoch_{}/tagged.json".format(model_root,epoch_num), "w") as f:
            f.write(objStr.encode('utf-8'))
    return data_lang_prec


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

def make_word_hashtable(data):
    yo = {}
    for w in data:
        if not w['word'] in yo:
            yo[w['word']] = set()
        yo[w['word']].add(w['tag'])
    return yo


# Split data to two sets with sizes: (percent_a %, (100 - percent_a) %)
def split_data(data, percent_a):
    split_index = int(round(len(data) * percent_a))
    a = data[:split_index]
    b = data[split_index:]

    return (a,b)

# read in all the data
all_data = list(read_data())

random.shuffle(all_data)
# train val will be split up 100-780

percent_training_validation = 0.7
percent_training = 0.6
train_val_data, test_data = split_data(all_data, percent_training_validation)
train_data, val_data = split_data(train_val_data, percent_training)

lang_hashtable = {}  # make_word_hashtable(train_data)

# create the vocabulary
let_vocab = Vocabulary()
lang_tags = ['aramaic','mishnaic','ambiguous']

# iterate through all the dapim and put everything in the vocabulary
for word in all_data:
    let_vocab.add_text(list(word['word']))

let_vocab.finalize()

lang_conf_matrix = ConfusionMatrix(len(lang_tags), lang_tags)

log_message('pos: ' + str(len(lang_tags)))
log_message('let: ' + str(let_vocab.size()))

# debug - write out the vocabularies
# write out to files the pos vocab and the letter vocab
with codecs.open('{}/let_vocab.txt'.format(model_root), 'w', encoding='utf8') as f:
    for let, id in let_vocab.get_c2i().items():
        f.write(str(id) + ' : ' + let + '\n')


# to save on memory space, we will clear out all_data from memory
all_data = None

# create the model and all it's parameters
model = dy.Model()

# create the word encoders (an encoder for the chars for the bilstm, and an encoder for the prev-pos lstm)
let_enc = WordEncoder("letenc", EMBED_DIM, model, let_vocab)

# the BiLSTM for all the chars, take input of embed dim, and output of the hidden_dim minus the embed_dim because we will concatenate
# with output from a separate bilstm of just the word
bilstm = BILSTMTransducer(BILSTM_LAYERS, EMBED_DIM, HIDDEN_DIM, model)

# now the class mlp, it will take input of 2*HIDDEN_DIM (A concatenate of the before and after the word) + EMBED_DIM from the prev-pos
# output of 2, unknown\talmud
lang_mlp = MLP(model, "classmlp", 2 * HIDDEN_DIM, HIDDEN_DIM, 3)


# the trainer
trainer = dy.AdamTrainer(model)

# if we are loading in a model
if filename_to_load:
    model.load(filename_to_load)

train_test = True
if train_test:
    lang_conf_matrix.clear()
    current_epoch_validation_accuracy = 0.0
    best_validation_accuracy = 0.0
    best_validation_accuracy_epoch_ind = -1
    best_validation_filename = None
    early_stop_counter = 0
    prev_epoch_validation_accuracy = run_network_on_validation(START_EPOCH - 1)

    # train!
    for epoch in range(START_EPOCH, 100):
        last_loss, last_lang_prec = 0.0, 0.0
        total_loss, total_lang_prec = 0.0, 0.0
        total_lang_items = 0

        # shuffle the train data
        random.shuffle(train_data)

        items_seen = 0
        # iterate
        for word_obj in train_data:
            # calculate the loss & prec
            loss, lang_prec = CalculateLossForWord(word_obj, fValidation=False)

            # forward propagate
            total_loss += loss.value() if loss else 0.0
            # back propagate
            if loss: loss.backward()
            trainer.update()

            # increment the prec variable
            total_lang_prec += lang_prec
            total_lang_items += 1

            items_seen += 1
            # breakpoint?
            breakpoint = 5000
            if items_seen % breakpoint == 0:
                last_loss = total_loss / breakpoint
                last_lang_prec = total_lang_prec / total_lang_items * 100

                log_message("Words processed: " + str(items_seen) + ", loss: " + str(last_loss) + ', lang_prec: ' + str(
                    last_lang_prec))

                total_loss, total_lang_prec = 0.0, 0.0
                total_lang_items = 0

        log_message('Finished epoch ' + str(epoch))
        prev_epoch_validation_accuracy = current_epoch_validation_accuracy
        current_epoch_validation_accuracy = run_network_on_validation(epoch)

        util.make_folder_if_need_be('{}/epoch_{}'.format(model_root, epoch))
        filename_to_save = '{}/epoch_{}/postagger_model_embdim{}_hiddim{}_lyr{}_e{}_trainloss{}_valacc{}.model'.format(
            model_root,epoch,EMBED_DIM,HIDDEN_DIM,sLAYERS,epoch,last_loss,current_epoch_validation_accuracy)
        model.save(filename_to_save)

        if current_epoch_validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = current_epoch_validation_accuracy
            best_validation_accuracy_epoch_ind = epoch
            best_validation_filename = filename_to_save
            early_stop_counter = 0
            log_message("Best validation loss so far!\n")

        else:
            if current_epoch_validation_accuracy < prev_epoch_validation_accuracy:
                early_stop_counter += 1
                log_message("Validation loss hasn't improved. Patience: {}/{}\n".format(early_stop_counter, EARLY_STOP_PATIENCE_N_EPOCHS))
            else:
                log_message("Validation loss improved. Resetting patience to 0\n")

        f = open("{}/epoch_{}/conf_matrix_e{}.html".format(model_root,epoch, epoch), 'w')
        f.write(lang_conf_matrix.to_html())
        f.close()
        lang_conf_matrix.clear()
        if early_stop_counter >= EARLY_STOP_PATIENCE_N_EPOCHS:
            break

    log_message('Epoch: {} with highest validation accuracy: {}'.format(best_validation_accuracy_epoch_ind, best_validation_accuracy))
    model.populate(best_validation_filename)
    test_acc = run_network_on_test()

    # Symlink to the best model
    import subprocess
    cmd = 'ln -s {} {}/best_val_{:.2f}'.format(best_validation_filename, model_root, test_acc)
    symlink_out = subprocess.check_output(cmd.split())
    log_message(symlink_out)

else:
    #tag all of shas!
    lang_tagged_path = 'data/3_lang_tagged'
    mesechtot_names = ['Berakhot','Shabbat','Eruvin','Pesachim','Bava Kamma','Bava Metzia','Bava Batra']
    for mesechta in mesechtot_names:
        mesechta_path = 'data/2_matched_sefaria/json/{}'.format(mesechta)


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
            cal_matcher_out = json.load(codecs.open('{}/{}'.format(mesechta_path,f_name), "rb", encoding="utf-8"))
            for w in cal_matcher_out['words']:
                lang, confidence = CalculateLossForWord(w, fRunning=True)
                lang_out.append({'word':w['word'],'lang':lang,'confidence':confidence})

            util.make_folder_if_need_be("{}/json/{}".format(lang_tagged_path,mesechta))
            fp = codecs.open("{}/json/{}/{}".format(lang_tagged_path,mesechta,f_name), "wb", encoding='utf-8')
            json.dump(lang_out, fp, indent=4, encoding='utf-8', ensure_ascii=False)
            fp.close()

            daf = f_name.split('/')[-1].split('.json')[0]
            html_out[daf] = lang_out
            if i_f % 10 == 0:
                print '{}/{}'.format(mesechta,f_name)
                html = print_tagged_corpus_to_html_table(html_out)
                util.make_folder_if_need_be("{}/html/{}".format(lang_tagged_path, mesechta))
                fp = codecs.open("{}/html/{}/{}.html".format(lang_tagged_path,mesechta, daf), "wb",
                                 encoding='utf-8')
                fp.write(html)
                fp.close()
                html_out = OrderedDict()


def run_experiment(method, n_repetitions):
    times, results = timereturn(method, n_repetitions, reset_weights)
    for i, res in enumerate(results):
        res['time'] = times[i]

    experiment = {'title' : '{}_X_{}'.format(method.func_name, n_repetitions), 'results': results}
    with open("{}.json".format(experiment['title']), 'w') as res_file:
        json.dump(experiment, res_file)

