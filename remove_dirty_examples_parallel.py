# -*- coding: utf-8 -*-
"""
pre-processing large monolingual corpus with several heuristics
"""
import argparse
import os
import re
import string
from pathlib import Path

import logging

SYMBOLS = set(string.punctuation)
ASCII_CHARS = set(string.printable)


def get_args():
    parser = argparse.ArgumentParser(description='my script')
    parser.add_argument('--inpsrc', '-is', default=None, help='files to read, if empty, stdin is used')
    parser.add_argument('--outsrc', '-os', required=True,help='path to output dir')
    parser.add_argument('--inptrg', '-it', required=True,help='path to output dir')
    parser.add_argument('--outtrg', '-ot', required=True, help='path to output dir')
    args = parser.parse_args()
    return args


def remove_long_sent(line, threshold=80):
    tokens = line.split(' ')
    if len(tokens) > threshold:
        return None
    else:
        return line


def remove_short_sent(line, threshold=3):
    tokens = line.split(' ')
    if len(tokens) <= threshold:
        return None
    else:
        return line


def remove_too_many_puncts(line, thresh_ratio=0.20):
    tokens = line.split(' ')
    n_puncs = len([t for t in tokens if t in SYMBOLS])
    n_total = len(tokens)
    ratio = (n_puncs / n_total)
    if ratio >= thresh_ratio and n_total >= 10:
        return None
    else:
        return line


def remove_nonascii_chars(line):
    filterred = ''.join(c for c in line if c in ASCII_CHARS)
    if len(filterred) < len(line):
        return None
    else:
        return line


def remove_consecutive_whitespace(line):
    if re.search(r'\s{3,}', line):
        return None
    else:
        return line


def remove_too_many_digits_sentence(line):
    total_tokens = len(line.split())
    match = re.findall(r'\s\d[\d,\/]*\s', line)
    if not match:
        return line
    else:
        n_digit_tokens = len(match)
        if n_digit_tokens / total_tokens > 0.10:
            return None
        else:
            return line
def detokenize(sent):
    '''restores a bpe-segmented sentence
        '''
    return sent.replace(" ", "").replace("▁", " ").strip()

def main(args):
    #dest = Path(args.output, *Path(args.input).parts[-1:])
    logging.info('Processing: {}'.format(args.inptrg))

    with open(args.inpsrc, 'r', encoding='utf-8') as isf, open(args.inptrg, 'r', encoding='utf-8') as itf, open(args.outsrc, 'w', encoding='utf-8') as osf, open(args.outtrg, 'w',encoding='utf-8') as otf:
        for tgt_line, src_line in zip(isf, itf):
            #tgt_line = tgt_line.strip()
            tgt_dtok = detokenize(tgt_line)
            # remove if tgt_dtok is too long
            if tgt_dtok:
                tgt_dtok = remove_long_sent(tgt_dtok)

            # remove if tgt_dtok is too short
            if tgt_dtok:
                tgt_dtok = remove_short_sent(tgt_dtok)

            # remove if certain ratio of tokens are symbols
            if tgt_dtok:
                tgt_dtok = remove_too_many_puncts(tgt_dtok)

            # remove if tgt_dtok contains non-ascii characters
            if tgt_dtok:
                tgt_dtok = remove_nonascii_chars(tgt_dtok)

            # remove if consecutive spaces exist (probably the numerical table)
            if tgt_dtok:
                tgt_dtok = remove_consecutive_whitespace(tgt_dtok)

            # remove if too many digits exist
            if tgt_dtok:
                tgt_dtok = remove_too_many_digits_sentence(tgt_dtok)

            if tgt_dtok:
                #tgt_line = tgt_line + '\n'
                #src_line = src_line + '\n'
                # fo.write(tgt_line.encode('utf-8'))
                otf.write(tgt_line)
                osf.write(src_line)




if __name__ == "__main__":
    args = get_args()
    main(args)
