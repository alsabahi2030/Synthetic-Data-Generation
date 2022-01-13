import logging
import os
from glob import glob
import argparse
from gec import filepath, word_tokenize
from gec import perturb2021 as perturb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def maybe_do(fp, func, inputs):
    if os.path.exists(fp):
        logging.info(f"skip this step as {fp} already exists")
    else:
        func(*inputs)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 1. word-tokenize
    parser.add_argument("--max_tokens", type=int, default=160,
                        help="Maximum number of tokens in a sample")

    # 2. train bpe model
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="vocabulary size")

    # 3. perturbation -> bpe-tokenize
    parser.add_argument("--min_cnt", type=int, default=4)
    parser.add_argument("--word_change_prob", type=float, default=.7)
    parser.add_argument("--type_change_prob", type=float, nargs="+", default=[0.3,0.07])
    parser.add_argument("--max_word_len", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, nargs="+", default=[1, 12, 5],
                        help="list of n_epochs of gutenberg, tatoeba, and wiki103")

    args = parser.parse_args()
    #args.word_change_prob = [0.8 , ]
    max_tokens_len = 25
    max_string_len = 300

    #args.type_change_prob = [0.4 / args.max_input_len, 0.3 / args.max_tokens, 0.2 / args.max_tokens, [0.2 / args.max_tokens,0.1 / args.max_tokens,0.1 / args.max_tokens]]
    args.type_change_prob = [0.5 / max_tokens_len, 0.5 / max_tokens_len, 0.3 / max_tokens_len,[0.3 / max_tokens_len, 0.3 / max_string_len, 0.2 / max_string_len]]
    #args.type_change_prob = [0.3, 0.07,0.05]
    fp = filepath.FilePath()
    fp.make_dirs()


    logging.info("STEP 1. Word-tokenize the original files and merge them")

    raw_data_path = "./data/raw"
    clean_data_path = "./data/clean"
    tok_data_path = "./data/tok"
    parallel_data_path = "./data/parallel"

    raw_files = os.listdir(raw_data_path)

    os.makedirs(tok_data_path,exist_ok=True)
    os.makedirs(clean_data_path,exist_ok=True)
    os.makedirs(parallel_data_path,exist_ok=True)

    logging.info("STEP 1. writing perturbation scenario")

    dir = f"{fp.wi_m2}/*train*.m2"
    word2ptbs = perturb.make_word2ptbs(sorted(glob(dir)), args.min_cnt)

    logging.info("STEP 2. clean and Word-tokenize the original files")
    for f in raw_files:
        f_tok = os.path.join(raw_data_path,f"{f}.tok")
        f_clean = os.path.join(tok_data_path,f"{f}.clean")
        if os.path.exists(f_clean):
            logging.info(f"skip this step as {f_clean} already exists")
        else:
            prompt = f"python3.6 remove_dirty_examples.py --input {os.path.join(raw_data_path,f)} --output {f_clean}"
            os.system(prompt)
        maybe_do(f_tok, word_tokenize.generalfiletokinzer,
                 (f_clean, f_tok, args.max_tokens))
        print(f"Done the file {f}")

    logging.info("STEP 3 Noise the original files")
    files= os.listdir(tok_data_path)
    for f in files:
        maybe_do(f"{parallel_data_path}{f}.ori", perturb.do,
                 (word2ptbs, os.path.join(tok_data_path,f),
                  f"{parallel_data_path}{f}.ori", f"{parallel_data_path}{f}.cor", args.n_epochs[0],
                  args.word_change_prob, args.type_change_prob))

