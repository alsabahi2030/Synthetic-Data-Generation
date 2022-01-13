# encoding: utf-8
'''
Created on Nov 26, 2015

@author: tal

Based in part on:
Learn math - https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py

See https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm
'''

from __future__ import print_function, division, unicode_literals
from hashlib import sha256
import json
import logging
import os
#from preprocessing import read_top_chars,analyze_ngram_word,preprocesses_data_filter,read_filtered_data,preprocesses_split_lines,preprocess_partition_data,analyze_ngrams,download_the_news_data,uncompress_data,clean_text,preprocesses_data_clean,preprocesses_data_analyze_chars,hasNumbers,analyze_ngrams,build_vocabulary
from gec.error_patterns import remove_double_char,swap_vowels,findt2false,word2homophone,w2phonetics_from_dict,letters_patterns,similar_sound_letters_1,similar_sound_letters_2,similar_double_sound_letters,end_with
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, \
    seed as random_seed, rand
from multiprocessing import Pool
from nltk import word_tokenize
#import preprocessing
from sacremoses import MosesTokenizer, MosesDetokenizer
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
#detokenizer = MosesDetokenizer()

# Set a logger for the module
LOGGER = logging.getLogger(__name__)  # Every log will use the module name
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.DEBUG)

CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
no_errors_type1=0
no_errors_type2=0
no_errors_type3=0
def inject_error_type1(type_of_error_fun,tokenized_words):
    #tokenized_words = word_tokenize(sent)
    #print('noise {} begin'.format(type_of_error_fun))
    for i, word in enumerate(tokenized_words):
        if word[0].isupper() or hasNumbers(word) or len(word) < 3:
            continue
        else:
            misspelled_word = type_of_error_fun(word)
            if misspelled_word != word:
                tokenized_words = tokenized_words[:i] + [misspelled_word] + tokenized_words[i + 1:]
    #sent = " ".join(tokenized_words)
    sent = md.detokenize(tokenized_words)
    #print('noise {} end'.format(type_of_error_fun))
    return sent

def inject_error_type1_2(type_of_error_fun,tokenized_words,amount_of_noise):
    #tokenized_words = word_tokenize(sent)
    #print('noise {} begin'.format(type_of_error_fun))
    try:
        n = amount_of_noise[0] * len(tokenized_words)
        n = int(n * len(tokenized_words))
        for i in range(n):
            random_word_position = random_randint(len(tokenized_words)-1)
            word = tokenized_words[random_word_position]
            if word[0].isupper() or hasNumbers(word) or len(word) < 3:
                continue
            else:
                misspelled_word = type_of_error_fun(word)
                if misspelled_word != word:
                    if random_word_position == len(tokenized_words)-1:
                        tokenized_words = tokenized_words[:random_word_position - 1] + [misspelled_word]
                    else:
                        tokenized_words = tokenized_words[:random_word_position] + [misspelled_word] + tokenized_words[random_word_position + 1:]
        #sent = " ".join(tokenized_words)
        sent = md.detokenize(tokenized_words)
    except ValueError:
        print("Oops! There was a problem in this sentence {}".format(tokenized_words))
    #print('noise {} end'.format(type_of_error_fun))
    return sent

def inject_error_type2(type_of_error_fun,sent):
    try:
        tokenized_words = mt.tokenize(sent)
        tokenized_words=['"' if w == '``' or w == "''" else w for w in tokenized_words]
        #print('noise {} begin'.format(type_of_error_fun))
        random_word_position = random_randint(len(tokenized_words)-1)
        i = 0
        while tokenized_words[random_word_position][0].isupper() or hasNumbers(
                tokenized_words[random_word_position]) or len(tokenized_words[random_word_position]) < 3:
            random_word_position = random_randint(len(tokenized_words))
            i += 1
            if i > 9:
                break
        if i < 9:
            misspelled_word = type_of_error_fun(tokenized_words[random_word_position])
            if misspelled_word != tokenized_words[random_word_position]:
                tokenized_words2 = tokenized_words[:random_word_position] + [misspelled_word] + tokenized_words[
                                                                                                random_word_position + 1:]
                #sent = " ".join(tokenized_words2)
                sent = md.detokenize(tokenized_words2)
        #print('noise {} end'.format(type_of_error_fun))
    except ValueError:
        print("Oops! There was a problem in this sentence {}".format(tokenized_words))
    return sent

def add_noise_to_string(a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    try:
        #tokenized_words = word_tokenize(a_string)
        tokenized_words = mt.tokenize(a_string)
        tokenized_words=['"' if w == '``' or w == "''" else w for w in tokenized_words]
        if rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1(findt2false, tokenized_words)
            global no_errors_type1
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1(word2homophone, tokenized_words)
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1(letters_patterns, tokenized_words)
            #global no_errors_type1
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1(swap_vowels, tokenized_words)
            #global no_errors_type1
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1(remove_double_char, tokenized_words)
            #global no_errors_type1
            no_errors_type1 += 1
        #################################################
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1_2(findt2false, tokenized_words,amount_of_noise)
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1_2(word2homophone, tokenized_words,amount_of_noise)
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1_2(letters_patterns, tokenized_words,amount_of_noise)
            #global no_errors_type1
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1_2(swap_vowels, tokenized_words,amount_of_noise)
            #global no_errors_type1
            no_errors_type1 += 1
        elif rand() < amount_of_noise[0] * len(tokenized_words):
            a_string = inject_error_type1_2(remove_double_char, tokenized_words,amount_of_noise)
            #global no_errors_type1
            no_errors_type1 += 1
        #################################################
        elif rand() < amount_of_noise[1] * len(a_string):
            a_string = inject_error_type2(similar_sound_letters_1, a_string)
            global no_errors_type2
            no_errors_type2 += 1
        elif rand() < amount_of_noise[1] * len(a_string):
            a_string = inject_error_type2(similar_sound_letters_2, a_string)
            no_errors_type2 += 1
        elif rand() < amount_of_noise[1] * len(a_string):
            a_string = inject_error_type2(similar_double_sound_letters, a_string)
            no_errors_type2 += 1
        elif rand() < amount_of_noise[1] * len(a_string):
            a_string = inject_error_type2(end_with, a_string)
            no_errors_type2 += 1
        elif rand() < amount_of_noise[1] * len(a_string):
            a_string = inject_error_type2(w2phonetics_from_dict, a_string)
            #global no_errors_type2
            no_errors_type2 += 1
        elif rand() < amount_of_noise[2] * len(a_string):
            # Replace a character with a random character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
            global no_errors_type3
            no_errors_type3 += 1
        elif rand() < amount_of_noise[2] * len(a_string):
            # Delete a character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
            #global no_errors_type3
            no_errors_type3 += 1
        elif rand() < amount_of_noise[2] * len(a_string):
            # Add a random character
            random_char_position = random_randint(len(a_string))
            a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
            #global no_errors_type3
            no_errors_type3 += 1
        elif rand() < amount_of_noise[2] * len(a_string):
            # Transpose 2 characters
            random_char_position = random_randint(len(a_string) - 1)
            a_string = (a_string[:random_char_position] + a_string[+ + 1] + a_string[
                random_char_position] +
                        a_string[random_char_position + 2:])
            #global no_errors_type3
            no_errors_type3 += 1
    except (ValueError):
        print("Oops! There was a problem in this sentence {}".format(a_string))
    return a_string

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

