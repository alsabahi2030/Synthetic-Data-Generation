
from nltk import ngrams
import os
import re
import pickle
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, \
    seed as random_seed, rand

silent_letters_patterns={'mb': 'm',
 'bt': 't',
 'tch': 'ch',
 'tm': 'm',
 'stle': 'sle',
 'wh': 'w',
 'hono': 'ono',
 'hou': 'ou',
 'hones': 'ones',
 'rh': 'r',
 'kn': 'n',
 'sw': 's',
 'wr': 'r',
 'who': 'ho',
 'gn': 'n',
 'gu': 'g',
 'ui': 'i',
 'sc': 's',
 'al': 'a',
 'pn': 'n',
 'ps': 's',
 'pb': 'b',
 'dg': 'g',
 'dn': 'n',
 'mn': 'm',
 'isl': 'il',
 'ough':'uf',
 'through':'thro',
 'though':'tho',
 'ea':'ae',
 'ei':'ie',
'au':'ua',
  'exh':'ex',
'tion':'sion',
'sion':'tion',
'sure':'shure',
'cture':'cshre',
'ere':'ear',
'ear':'ere'}

vowels_combinations=['ea','ou','ei','ie','ai','uo','io','oi','au','ua','ow','wo']

vowels_dict={'a':'u','e':'i','o':'u','f':'v','w':'o'}
vowels=['a','e','i','o','u']
remove_final_letters = ['e','er','ment','ly','s','est','ed','ing','ful']
# pylint:disable=invalid-name

DATA_FILES_PATH = "/data_local/src/helo_word-master/resources/"
#DATA_FILES_FULL_PATH = os.path.expanduser(os.path.join(DATA_FILES_PATH,'.'))
DATA_FILES_FULL_PATH = DATA_FILES_PATH
"""
DATA_FILES_PATH = "/container_data/data/deepspelling2/chunkrawdata"
DATA_FILES_FULL_PATH = os.path.expanduser(os.path.join(DATA_FILES_PATH,'..'))
"""
DATA_FILES_URL = "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
NEWS_FILE_NAME_COMPRESSED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.shuffled.gz")  # 1.1 GB
NEWS_FILE_NAME_ENGLISH = "news.2013.en.shuffled"
NEWS_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, NEWS_FILE_NAME_ENGLISH)
NEWS_FILE_NAME_CLEAN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.clean")
NEWS_FILE_NAME_FILTERED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.filtered")
NEWS_FILE_NAME_SPLIT = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split.small")
NEWS_FILE_NAME_SPLIT_SHUFFLED = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.split.shuffled")
# NEWS_FILE_NAME_TRAIN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.train")
NEWS_FILE_NAME_TRAIN = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.validate")
NEWS_FILE_NAME_VALIDATE = os.path.join(DATA_FILES_FULL_PATH, "news.2013.en.validate")
CHAR_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "char_frequency.json")
BIGRAMS_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "bigrams_frequency.json")
TRIGRAMS_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "trigrams_frequency.json")
VOCABS_FREQUENCY_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "vocabs_frequency.json")
SAVED_MODEL_FILE_NAME = os.path.join(DATA_FILES_FULL_PATH, "keras_spell_e{}.h5")  # an HDF5 file

NEWS_FILE_NAME_SOURCE = os.path.join(DATA_FILES_PATH, "processed/news.2013.en.sr")
NEWS_FILE_NAME_TARGET = os.path.join(DATA_FILES_PATH, "processed/news.2013.en.tar")

char_set1 = ['a', 'e', 'o', 'f', 'w', 'u', 'i', 'u', 'v', 'o','i','y']
char_set2 = ['u', 'i', 'u', 'v', 'o', 'a', 'e', 'o', 'f', 'w','y','i']
similar_sound_dict={'a':['u'], 'b':['p'], 'p':['b'], 'e':['i','a'], 'o':['u','w'], 'f':['v'], 'w':['o','u'], 'u':['a','o','w'], 'i':['e','a','y'],'v':['f'],'y':['i']}

def getting_ngrams(word, n):
    lngrams=list(ngrams(word, n))
    listofngrams=[''.join(g) for g in lngrams]
    return listofngrams

def remove_double_char(word):
    new_word=word
    pattern = re.compile(r'(\w)\1*')
    result = [x.group() for x in pattern.finditer(word)]
    double_letters = [w for w in result if len(w) == 2]
    if len(double_letters) == 1:
        new_word = new_word.replace(double_letters[0], double_letters[0][1] + double_letters[0][0])
    elif len(double_letters) > 1:
        random_number_change = random_randint(1,len(double_letters))
        random_shuffle(double_letters)
        for i in range(random_number_change):
            new_word = new_word.replace(double_letters[i], double_letters[i][0])
    return new_word

def swap_vowels(word):
    new_word = word
    vowel=[v for v in  vowels_combinations if v in  word]
    if len(vowel) == 1:
        new_word = new_word.replace(vowel[0], vowel[0][1] + vowel[0][0])
    elif len(vowel) > 1:
        random_number_change = random_randint(1,len(vowel))
        random_shuffle(vowel)
        for i in range(random_number_change):
            new_word = new_word.replace(vowel[i], vowel[i][1]+vowel[i][0])
    return new_word

def findt2false(word):
    new_word=word
    temp = open(os.path.join(DATA_FILES_FULL_PATH,'t2false3.pkl'), 'rb')
    t2false = pickle.load(temp)
    if new_word in t2false:
        if len(t2false[new_word]) > 1:
            random_word_position = random_randint(len(t2false[new_word]))
            return t2false[new_word][random_word_position]
        return t2false[new_word][0]
    #print('{} ==> {}'.format(word, new_word))
    return new_word

def word2homophone(word):
    new_word=word
    temp = open(os.path.join(DATA_FILES_FULL_PATH,'common_homophones_dict.pkl'), 'rb')
    w2homophone = pickle.load(temp)
    if new_word in w2homophone:
        if len(w2homophone[new_word]) > 1:
            random_word_position = random_randint(len(w2homophone[new_word]))
            return w2homophone[new_word][random_word_position]
        return w2homophone[new_word][0]
    #print('{} ==> {}'.format(word, new_word))
    return new_word

def w2phonetics_from_dict(word):
    try:
        new_word=word
        phonetics_dictionary = pickle.load(open(os.path.join(DATA_FILES_FULL_PATH, 'big_list_ph_dictionary.pkl'), 'rb'))
        words2phonetics = pickle.load(open(os.path.join(DATA_FILES_FULL_PATH, 'big_words2phonetics_dict.pkl'), "rb"))
        ph = words2phonetics[new_word]
        if not ph:
            return word
        ph_dist0 = phonetics_dictionary[ph]
        if not ph_dist0:
            return word
        elif len(ph_dist0)> 1:
            random_word_position = random_randint(len(ph_dist0))
            i=0
            while ph_dist0[random_word_position].lower() == new_word:
                random_word_position = random_randint(len(ph_dist0))
                i+=1
                if i>5:
                    break
        else:
            random_word_position=0
        #print('{} ==> {}'.format(word, new_word))
        return ph_dist0[random_word_position].lower()
    except KeyError:
        return word

def letters_patterns(word):
    new_word=word
    sl = [w for w in silent_letters_patterns if w in new_word]
    if len(sl) == 1:
        new_word = new_word.replace(sl[0],silent_letters_patterns[sl[0]])

    elif len(sl) > 1:
        random_number_change = random_randint(1,len(sl))
        random_shuffle(sl)
        for i in range(random_number_change):
            new_word = new_word.replace(sl[i], silent_letters_patterns[sl[i]])
    return new_word

def similar_double_sound_letters(word):
    new_word=word
    pattern = re.compile(r'(\w)\1*')
    result = [x.group() for x in pattern.finditer(new_word)]
    double_letters = [w for w in result if len(w) == 2 and w[0] in similar_sound_dict.keys()]
    if len(double_letters) == 1:
        random_sound= random_randint(len(similar_sound_dict[double_letters[0][0]]))
        new_word = new_word.replace(double_letters[0],similar_sound_dict[double_letters[0][0]][random_sound])
    elif len(double_letters) >1 :
        random_number_change = random_randint(1,len(double_letters))
        random_shuffle(double_letters)
        for i in range(random_number_change):
            random_sound = random_randint(len(similar_sound_dict[double_letters[i][0]]))
            new_word = new_word.replace(double_letters[i],similar_sound_dict[double_letters[i][0]][random_sound])
    return new_word

def similar_sound_letters_1(word):
    new_word=word
    similar_sounds = [c for c in similar_sound_dict.keys() if c in new_word]
    if len(similar_sounds) == 1:
        random_sound = random_randint(len(similar_sound_dict[similar_sounds[0]]))
        new_word = new_word.replace(similar_sounds[0],similar_sound_dict[similar_sounds[0]][random_sound])
    elif len(similar_sounds) > 1:
        random_number_change = random_randint(1,len(similar_sounds))
        random_shuffle(similar_sounds)
        for i in range(random_number_change):
            random_sound = random_randint(len(similar_sound_dict[similar_sounds[i]]))
            new_word = new_word.replace(similar_sounds[i], similar_sound_dict[similar_sounds[i]][random_sound])
    return new_word

def similar_sound_letters_2(word):
    new_word=word
    pattern = re.compile(r'(\w)\1*')
    result = [x.group() for x in pattern.finditer(new_word)]
    similar_sounds=[c for c in result if c in char_set1]
    double_letters = [w for w in result if len(w) == 2 and w[0] in similar_sound_dict.keys()]
    random_error = random_randint(2)
    if random_error==1 and double_letters  :
        random_double_position= random_randint(len(double_letters))
        random_sound= random_randint(len(similar_sound_dict[double_letters[random_double_position][0]]))
        new_word = new_word.replace(double_letters[random_double_position],similar_sound_dict[double_letters[random_double_position][0]][random_sound])
    elif similar_sounds:
        random_position= random_randint(len(similar_sounds))
        random_sound= random_randint(len(similar_sound_dict[similar_sounds[random_position]]))
        new_word = new_word.replace(similar_sounds[random_position],similar_sound_dict[similar_sounds[random_position]][random_sound])
    return new_word

def end_with(word):
    chr = [c for c in remove_final_letters if word.endswith(c)]
    if chr:
        word = word.replace(chr[0],'')
        #print("Some characters are removed {} from {} ".format(chr, word))
    return word
