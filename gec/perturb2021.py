import random
import re, os
#import sentencepiece as spm
from tqdm import tqdm
from collections import Counter
from gec.noise_data import NoiseInjector
from gec.CorruptingTheDatase import add_noise_to_string
from nltk import pos_tag
from word_forms.word_forms import get_word_forms
from pattern3.en import conjugate, pluralize, singularize
import logging
import multiprocessing
from gec.error_patterns import remove_double_char,swap_vowels,findt2false,word2homophone,w2phonetics_from_dict,letters_patterns,similar_sound_letters_1,similar_sound_letters_2,similar_double_sound_letters,end_with
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def transform_suffix(word, suffix_1, suffix_2):
    if len(suffix_1) > len(word):
        print("suffix: {} to be replaced has larger length than word: {}".format(suffix_1, word))
        return word

    l1 = len(suffix_1)
    l2 = len(suffix_2)

    if word[-l1:] == suffix_1:
        return word[:-l1] + suffix_2
    else:
        print("transform")
        print("**** WARNING: SUFFIX : {} NOT PRESENT in WORD: {} ****".format(suffix_1, word))
        return word

def is_append_suffix(suffix,w1,w2):
  l = len(suffix)
  if (w2[-l:] == suffix) and (w1 == w2[:-l]):
    return True
  else:
    return False

def remove_er(self):
    if is_append_suffix("er",self.tgt_word,self.src_word):
        return "$SUFFIXTRANSFORM_REMOVE_er"
    else:
        return None

def remove_ed(self):
    if is_append_suffix("ed",self.tgt_word,self.src_word):
        return "$SUFFIXTRANSFORM_REMOVE_ed"
    else:
        return None
def append_ed(self):
    if is_append_suffix("ed",self.src_word, self.tgt_word):
        return "$SUFFIXTRANSFORM_APPEND_ed"
    else:
        return None
def append_ing(self):
    if is_append_suffix("ing",self.src_word, self.tgt_word):
        return "$SUFFIXTRANSFORM_APPEND_ing"
    else:
        return None

def remove_ing(self):
    if is_append_suffix("ing",self.tgt_word,self.src_word):
        return "$SUFFIXTRANSFORM_REMOVE_ing"
    else:
        return None
def remove_est(self):
    if is_append_suffix("est",self.tgt_word,self.src_word):
        return "$SUFFIXTRANSFORM_REMOVE_est"
    else:
        return None

def append_est(self):
    if is_append_suffix("est", self.src_word, self.tgt_word):
        return "$SUFFIXTRANSFORM_APPEND_est"
    else:
        return None

def append_er(self):
    if is_append_suffix("er",self.src_word, self.tgt_word):
        return "$SUFFIXTRANSFORM_APPEND_er"
    else:
        return None


def append_suffix(word, suffix):
    # print("word: {}, suffix: {}".format(word, suffix))

    if len(word) == 1:
        print(
            "We currently fear appending suffix: {} to a one letter word, but still doing it: {}".format(suffix, word))
        # return word

    l = len(suffix)

    # if word[-l:] == suffix:
    # print("**** WARNING: SUFFIX: {} ALREADY PRESENT in WORD, BUT still adding it: {} ****".format(suffix, word))

    if word[-1] == "s" and suffix == "s":
        return word + "es"

    if word[-1] == "y" and suffix == "s" and len(word) > 2:
        if word[-2] not in ["a", "e", "i", "o", "u"]:
            return word[0:-1] + "ies"

    # if word[-1] == "h" and suffix == "s":
    #    return word + "es"

    # if word[-1] == "t" and suffix == "d":
    #    return word + "ed"

    # if word[-1] == "k" and suffix == "d":
    #    return word + "ed"

    return word + suffix


def remove_suffix(word, suffix):
    if len(suffix) > len(word):
        print("suffix: {} to be removed has larger length than word: {}".format(suffix, word))
        return word

    l = len(suffix)

    if word[-l:] == suffix:
        return word[:-l]
    else:
        print("**** WARNING: SUFFIX : {} NOT PRESENT in WORD: {} ****".format(suffix, word))
        return word


def transform_suffix(word, suffix_1, suffix_2):
    if len(suffix_1) > len(word):
        print("suffix: {} to be replaced has larger length than word: {}".format(suffix_1, word))
        return word

    l1 = len(suffix_1)
    l2 = len(suffix_2)

    if word[-l1:] == suffix_1:
        return word[:-l1] + suffix_2
    else:
        print("transform")
        print("**** WARNING: SUFFIX : {} NOT PRESENT in WORD: {} ****".format(suffix_1, word))
        return word


PREPOSITIONS = [
    '', 'of', 'with', 'at', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout',
    'despite', 'towards', 'upon', 'concerning', 'to', 'in', 'for', 'on', 'by', 'about', 'like',
    'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within', 'along',
    'following', 'across', 'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down'
    'off', 'above', 'near']

VERB_TYPES = ['inf', '1sg', '2sg', '3sg', 'pl', 'part', 'p', '1sgp', '2sgp', '3sgp', 'ppl', 'ppart']
DET_TYPES = ['the', 'a', 'an', 'that', 'this','']
SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def change_type(word, tag):
    global PREPOSITIONS, VERB_TYPES
    if tag == "IN":
        #if random.random() < change_prob:
        word = random.choice(PREPOSITIONS)
    elif tag == "NN":
        #if random.random() < change_prob:
         word = pluralize(word)
    elif tag == "NNS":
        #if random.random() < change_prob:
        word = singularize(word)
    elif "VB" in tag:
        #if random.random() < change_prob:
        verb_type = random.choice(VERB_TYPES)
        word = conjugate(word, verb_type)
    elif "DT" in tag:
        #if random.random() < change_prob:
        word = random.choice(DET_TYPES)
    return word
def change_suffix(word,suffix1,suffix2):
    rc = random.choice([0, 1])
    if word.endswith(suffix1):
        if rc == 0:
            word = remove_suffix(word, suffix1)
        else:
            word = transform_suffix(word, suffix1, suffix2)
    return word

def change_type_random(word, tag):
    global PREPOSITIONS, VERB_TYPES
    try:
        if tag == "JJ":
            new_words = list(get_word_forms(word)['r'])
            if len(new_words) > 0:
                word = random.choice(new_words)
        elif tag == "JJR":
            word=change_suffix(word,'er','est')
        elif tag == "JJS":
            word=change_suffix(word,'est','er')
        elif tag == "RB":
            rc = random.choice([0, 1])
            if rc ==0:
                new_words = list(get_word_forms(word)['a'])
                if len(new_words) > 0:
                    word = random.choice(new_words)
            else:
                if word.endswith('ly'):
                    word = remove_suffix(word, 'ly')
                else:
                    word = append_suffix(word, 'ly')
        elif tag == "NN":
            new_words = list(get_word_forms(word)['a'])
            if len(new_words) > 0:
                word = random.choice(new_words)
        elif tag == "VB":
            new_words = list(get_word_forms(word)['v'])
            if len(new_words) > 0:
                word = random.choice(new_words)
    except:
        return word
    return word

def make_word2ptbs(m2_files, min_cnt):
    '''Error Simulation
    m2: string. m2 file path.
    min_cnt: int. minimum count
    '''
    word2ptbs = dict()  # ptb: pertubation
    for m2_file in m2_files:
        entries = open(m2_file, 'r',encoding='utf-8').read().strip().split("\n\n")
        for entry in entries:
            if entry:
                skip = ("noop", "UNK", "Um")
                S = entry.splitlines()[0][2:] + " </s>"
                words = S.split()
                edits = entry.splitlines()[1:]

                skip_indices = []
                for edit in edits:
                    features = edit[2:].split("|||")
                    if features[1] in skip: continue
                    start, end = features[0].split()
                    start, end = int(start), int(end)
                    word = features[2]

                    if start == end:  # insertion -> deletion
                        ptb = ""
                        if word in word2ptbs:
                            word2ptbs[word].append(ptb)
                        else:
                            word2ptbs[word] = [ptb]
                    elif start + 1 == end and word == "":  # deletion -> substitution
                        ptb = words[start] + " " + words[start + 1]
                        word = words[start + 1]
                        if word in word2ptbs:
                            word2ptbs[word].append(ptb)
                        else:
                            word2ptbs[word] = [ptb]
                        skip_indices.append(start)
                        skip_indices.append(start + 1)
                    elif start + 1 == end and word != "" and len(word.split()) == 1:  # substitution
                        ptb = words[start]
                        if word in word2ptbs:
                            word2ptbs[word].append(ptb)
                        else:
                            word2ptbs[word] = [ptb]
                        skip_indices.append(start)
                    else:
                        continue

                for idx, word in enumerate(words):
                    if idx in skip_indices: continue
                    if word in word2ptbs:
                        word2ptbs[word].append(word)
                    else:
                        word2ptbs[word] = [word]

    # pruning
    _word2ptbs = dict()
    for word, ptbs in word2ptbs.items():
        ptb2cnt = Counter(ptbs)

        ptb_cnt_li = []
        for ptb, cnt in ptb2cnt.most_common(len(ptb2cnt)):
            if cnt < min_cnt: break
            ptb_cnt_li.append((ptb, cnt))

        if len(ptb_cnt_li) == 0: continue
        if len(ptb_cnt_li) == 1 and ptb_cnt_li[0][0] == word: continue

        _ptbs = []
        for ptb, cnt in ptb_cnt_li:
            _ptbs.extend([ptb] * cnt)

        _word2ptbs[word] = _ptbs

    return _word2ptbs

def apply_perturbation(line, word2ptbs, word_change_prob, type_change_prob,noise_injector):
    words = line.split()
    word_tags = pos_tag(words)

    if random.random() < type_change_prob[0] * len(words):
        sent = []
        for (_, t), w in zip(word_tags, words):
            if not w[0].isupper():
                if w in word2ptbs and random.random() > 1-word_change_prob:
                    oris = word2ptbs[w]
                    w = random.choice(oris)
                elif t in ["IN","NN","NNS","VB","DT"] and random.random() < type_change_prob[0]:
                    w = change_type(w, t)
                elif t in ["NN","NNS","VB","JJ","JJR","JJS","RB"] and random.random() < type_change_prob[0]:
                    w = change_type_random(w, t)
            sent.append(w)
        try:
            sent = " ".join(sent) + '\n'
            sent = re.sub("[ ]+", " ", sent)
        except:
            return None
    elif random.random() < type_change_prob[1] * len(words):
        sent, _ = noise_injector.inject_noise(words)
        if sent is None:
            return line
        sent = " ".join(sent) + '\n'
    elif random.random() < type_change_prob[2] * len(line):
        sent = add_noise_to_string(line, type_change_prob[3])
    else:
        return line
    """
    print("------------------------------")
    print(f" Org Sent=> {line} ")
    print(f" New Sent=> {sent} ")
    print("------------------------------")
    """
    return sent

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def count_lines(f):
    with open(f, "r",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def make_parallel_single(word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob):
    #logging.info("Load sentencepiece model")
    #sp = spm.SentencePieceProcessor()
    #sp.Load(bpe_model)

    with open(ori, 'w',encoding='utf-8') as ori, open(cor, 'w',encoding='utf-8') as cor:
        for _ in tqdm(range(n_epochs)):
            i = 0
            lines = open(txt, 'r',encoding='utf-8').readlines()
            tgts = [tokenize_line(line.strip()) for line in lines]
            noise_injector = NoiseInjector(tgts)

            for line in lines:
                #words = line.strip().split()
                perturbation = apply_perturbation(line, word2ptbs, word_change_prob, type_change_prob, noise_injector)
                perturbation=perturbation.rstrip()
                if perturbation is None: continue
                #ori_pieces = sp.EncodeAsPieces(perturbation)
                #cor_pieces = sp.EncodeAsPieces(line)
                line = line.rstrip()
                ori.write(perturbation + "\n")
                cor.write(line + "\n")


def make_parallel(inputs):
    word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob, start, end = inputs
    #logging.info("Load sentencepiece model")
    #sp = spm.SentencePieceProcessor()
    #sp.Load(bpe_model)

    ori_dir = os.path.join(os.path.dirname(ori), f"working/ori")
    cor_dir = os.path.join(os.path.dirname(ori), f"working/cor")
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(cor_dir, exist_ok=True)

    with open(f'{ori_dir}/{start}', 'w',encoding='utf-8') as ori, open(f'{cor_dir}/{start}', 'w',encoding='utf-8') as cor:
        for _ in range(n_epochs):
            i = 0
            lines = open(txt, 'r',encoding='utf-8').readlines()
            tgts = [tokenize_line(line.strip()) for line in lines]
            noise_injector = NoiseInjector(tgts)

            for line in tqdm(lines):
                i += 1
                if start <= i < end:
                    #words = line.strip().split()
                    perturbation = apply_perturbation(line, word2ptbs, word_change_prob, type_change_prob,noise_injector)
                    perturbation = perturbation.rstrip()
                    if perturbation is None: continue
                    #ori_pieces = sp.EncodeAsPieces(perturbation)
                    #cor_pieces = sp.EncodeAsPieces(line)
                    line = line.rstrip()
                    ori.write(perturbation + "\n")
                    cor.write(line + "\n")
                    if i%500==0:
                        print(f" {i} record have been processed!")
                if i > end:
                    break

def do(word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob):
    print("# multiprocessing settings")
    n_cpus = multiprocessing.cpu_count()
    n_cpus = 30



    print(f"# Working on {txt}")
    n_lines = count_lines(txt)
    print(f"# prepare inputs{n_lines}")

    print(f"# {n_lines}inputs, {n_cpus}")
    start_li = list(range(0, n_lines, n_lines // n_cpus))
    start_end_li = [(start, start + n_lines // n_cpus) for start in start_li]
    inputs_li = [(word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob, start, end) \
                 for start, end in start_end_li]

    print("# work")
    single = False
    if single:
        make_parallel_single(word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob)
    else:
        p = multiprocessing.Pool(n_cpus)
        p.map(make_parallel, inputs_li)
        p.close()
        p.join()

    print("# work done!")

    print("# concat...")
    os.system(f"cat {os.path.dirname(ori)}/working/ori/* > {ori}")
    os.system(f"cat {os.path.dirname(cor)}/working/cor/* > {cor}")
    os.system(f"rm -r {os.path.dirname(ori)}/working")
    print("All done!")
