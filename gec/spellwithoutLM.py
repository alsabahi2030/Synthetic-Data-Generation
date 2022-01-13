import hunspell

import copy
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from lm_scorer import LMScorer
from .filepath import Path
from .fix_tokenization_errors import fix

verbose = False
NUM_OF_SUGGESTIONS = 1

SUGGESTION_CACHE = {}
CAPITAL_WORDS_DIC = {}

def suggest(text, is_first, speller, n=NUM_OF_SUGGESTIONS):
    if (text, is_first) in SUGGESTION_CACHE:
        return SUGGESTION_CACHE[(text, is_first)]

    if text.lower() in CAPITAL_WORDS_DIC:
        suggestions = [CAPITAL_WORDS_DIC[text.lower()]]

    else:
        suggestions = []

    is_upper = text[0].isupper()

    if speller.spell(text):
        SUGGESTION_CACHE[(text, is_first)] = suggestions
        return suggestions

    if text.isalpha():
        if is_upper:
            pass

        elif is_first:
            raw_suggestions = speller.suggest(text)
            for suggestion in raw_suggestions:
                suggestions.append(suggestion[0].upper() + suggestion[1:])

        else:
            raw_suggestions = speller.suggest(text)

            for suggestion in raw_suggestions:
                if suggestion[0].isupper():
                    continue
                suggestions.append(suggestion)

            if len(suggestions)==0:
                suggestions = []
            else:
                suggestions = suggestions[:n]

    if len(suggestions)==0:
        SUGGESTION_CACHE[(text, is_first)] = []
        return []
    else:
        returns = []
        for suggestion in suggestions:
            if '-' in suggestion:
                pass
            else:
                returns.append(suggestion)

        if len(returns) == 0:
            SUGGESTION_CACHE[(text, is_first)] = [text]
            return [text]
        else:
            SUGGESTION_CACHE[(text, is_first)] = returns
            return returns

def spellcheck(fin, fout, speller):

    with open(fout, 'w',encoding='utf-8',errors='ignore') as fout:
        lines = open(fin, 'r',encoding='utf-8',errors='ignore').read().splitlines()
        for line in tqdm(lines):
            corrected_sents = []
            sents = sent_tokenize(line)
            for sent in sents:
                sent = fix(sent)
                tokens = copy.deepcopy(sent).split()
                for i, word in enumerate(tokens):
                    if i==0:
                        is_first = True
                    else:
                        is_first = False

                    suggestions = suggest(word, is_first, speller)
                    if len(suggestions) == 0: continue
                    if len(suggestions) == 1:
                        if suggestions[0] == word: continue

                    candidates = []
                    for suggestion in suggestions:
                        tokens[i] = suggestion
                        candidates.append(" ".join(tokens))

                    # inspect
                    winner = ""

                    # For Neural Net
                    min_score = 1000

                    # scores = {idx: idx for idx in range(len(candidates))}

                    for idx, s in zip(range(len(candidates)), suggestions):
                        # print(s, score)
                        winner = s

                    if verbose:
                        print("ORIGINAL: " + sent)
                        print("TARGET: " + word.text)
                        print("SUGGEST: " + ','.join(suggestions))
                        for idx, candidates in zip(range(len(scores)), candidates):
                            print("{}: ".format(scores[idx]) + candidates)
                        print("WINNER: " + winner)
                        input()

                    tokens[i] = winner.replace(" ", "_")
                corrected_sents.append(" ".join(tokens).replace("_", " "))
            fout.write(" ".join(corrected_sents) + "\n")

def check(fin, fout,
         aff=Path.aff,
         dic=Path.dic,
         lm_path=Path.lm_path2,
         data_bin=Path.lm_databin2,
         cap_word_dic=Path.cap_word_dic
         ):

    speller = hunspell.HunSpell(dic, aff)

    spellcheck(fin, fout, speller=speller)
