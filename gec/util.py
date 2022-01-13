import os
import logging
import re
from glob import glob
from heapq import nlargest

def maybe_do(fp, func, inputs):
    if os.path.exists(fp):
        logging.info(f"skip this step as {fp} already exists")
    else:
        func(*inputs)


def maybe_prompt(fp, prompt):
    if os.path.exists(fp):
        logging.info(f"skip this step as {fp} already exists")
    else:
        os.system(prompt)


def get_scores(report_fname, scorer):
    assert scorer in ["errant", "m2scorer"]

    report = open(report_fname, 'r',encoding='utf-8')

    try:
        if scorer == "errant":
            line = report.read().strip().splitlines()[-2]
            tokens = line.split()
            precision, recall, fscore = tokens[-3], tokens[-2], tokens[-1]

        else:  # m2scorer
            line = report.read().strip().splitlines()
            precision = line[0].split()[-1]
            recall = line[1].split()[-1]
            fscore = line[2].split()[-1]
    except:
        logging.error(f"[Util] cannot get scores from {report_fname}")
        precision, recall, fscore = 0, 0, 0
    try:
        precision = float(precision) * 100
        recall = float(recall) * 100
        fscore = float(fscore) * 100
    except:
        logging.error(f"[Util] cannot get scores from {report_fname}")


    # precision = int(float(precision) * 100)
    # recall = int(float(recall) * 100)
    # fscore = int(float(fscore) * 100)

    return precision, recall, fscore


def find_highest_score(output_dir, ori_path, scorer_type):
    data_basename = get_basename(ori_path, include_path=False, include_extension=False)
    if scorer_type=='m2scorer':
        files = glob(os.path.join(output_dir, f"*{data_basename}*.m2report"))
    else:
        files = glob(os.path.join(output_dir, f"*{data_basename}*.report"))

    highest_fscore = 0
    highest_basename = ''

    for report_fname in files:
        precision, recall, fscore = get_scores(report_fname, scorer_type)
        if fscore > highest_fscore:
            highest_fscore  = fscore
            highest_basename = get_basename(report_fname, include_path=True, include_extension=False).replace(data_basename, '')[:-1]

    highest_ckpt = f"{highest_basename}.pt".replace("outputs", "ckpt")
    if highest_basename == '':
        print(f'cannot find highest basename from {output_dir}')
        exit()

    return highest_fscore, highest_ckpt



def find_n_highest_scores(output_dir, ori_path, scorer_type,n=5):
    data_basename = get_basename(ori_path, include_path=False, include_extension=False)
    if scorer_type=='m2scorer':
        files = glob(os.path.join(output_dir, f"*{data_basename}*.m2report"))
    else:
        files = glob(os.path.join(output_dir, f"*{data_basename}*.report"))

    #highest_n_fscore = []
    highest_basename = ''
    scores = []
    ckpt_files = []
    for report_fname in files:
        try:
            precision, recall, fscore = get_scores(report_fname, scorer_type)
            scores.append(fscore)
            ckpt_files.append(get_basename(report_fname, include_path=True, include_extension=False).replace(data_basename, '')[:-1])
        except TypeError:
            print("Oops! Something wrong {report_fname}")

    highest_n_fscore_idx = nlargest(n, range(len(scores)), key=lambda idx: scores[idx])
    highest_n_ckpt = [ckpt_files[f] for f in highest_n_fscore_idx]
    highest_n_fscores = [scores[i] for i in highest_n_fscore_idx]
    highest_ckpts=[]
    report_files=[]
    for f in highest_n_ckpt:
        highest_ckpts.append(f"{f}.pt".replace("outputs", "ckpt"))
        report_files.append(f)
    if highest_n_fscore_idx == '':
        print(f'cannot find highest basename from {output_dir}')
        exit()

    return highest_n_fscores, highest_ckpts,report_files



def get_sorted_ckpts(ckpt_dir, epoch_start=1, epoch_end=5000, epoch_interval=1, bestckpt=False):
    files = glob(os.path.join(ckpt_dir, "*pt"))
    if bestckpt:
        return  files
    epoch_files = []
    for f in files:
        epoch = f.split("/")[-1].split(".")[0].replace("checkpoint", "").split("_")[-1]
        try:
            epoch = int(epoch)
        except ValueError:
            continue
        epoch_files.append((epoch, f))
    epoch_files = sorted(epoch_files, key=lambda x: x[0])
    files = [f for epoch, f in epoch_files if epoch_start <= epoch <= epoch_end]
    files = files[::epoch_interval]  # skip some

    return files


def get_basename(path, include_path=True, include_extension=True):
    if path is None:
        return None

    if os.path.isdir(path) and path[-1] == '/':
        path = path[:-1]
    base = os.path.basename(path)

    if os.path.isfile(path) and not include_extension:
        base = '.'.join(base.split('.')[:-1])

    if include_path:
        dirpath = os.path.dirname(path)
        return f'{dirpath}/{base}'
    else:
        return base


def get_cor_path(system_out, remove_unk_edits, remove_error_type_lst,
                 apply_rerank, preserve_spell, max_edits):
    cor_path = get_basename(system_out, include_extension=False)
    if remove_unk_edits:
        cor_path += '-un'
    if len(remove_error_type_lst) > 0:
        cor_path += '-'.join(remove_error_type_lst)
    if apply_rerank:
        cor_path += '-re'
    if preserve_spell:
        cor_path += '-sp'
    if max_edits is not None:
        cor_path += f'-m-{max_edits}'
    return f"{cor_path}.cor"


def change_ckpt_dir(ckpt_fpath, new_ckpt_dir):
    fpath_basename = os.path.basename(ckpt_fpath)
    dirname = os.path.dirname(ckpt_fpath).split('/')[:-1]
    new_ckpt_basename = os.path.basename(new_ckpt_dir)
    dirname = '/'.join(dirname + [new_ckpt_basename])
    return f"{dirname}/{fpath_basename}"

def shorten_name(path):
    base_dir = os.path.dirname(path).split('/')
    path_base = get_basename(path, include_path=False, include_extension=True)
    path_base = path_base.replace('_bpe60-', '6').replace('_bpe50-', '5').replace('_bpe60_', '6').replace(
        '_bpe50_', '5').replace('_', '')
    path = '/'.join(base_dir + [path_base])
    return path
