import os
import sys
import numpy as np
import torch
import random
import logging

class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        return self.total / self.num

def setup_dirs(output_dir):
    log_dir = os.path.join(output_dir, 'logs')
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    pred_dir = os.path.join(output_dir, 'preds')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return log_dir, ckpt_dir, pred_dir

def setup_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def set_logger(log_path, log_name='training'):
    if log_path is None:
        print('log_path is empty')
        return None
        
    if os.path.exists(log_path):
        print('%s already exists'%log_path)
        return None

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    logfile = logging.FileHandler(log_path)
    console = logging.StreamHandler()
    logfile.setLevel(logging.INFO)
    logfile.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(logfile)
    logger.addHandler(console)
    
    logger.propagate = False
    return logger

def evaluate_caption(ref_caps=None, pred_caps=None, scorer_names=None, remove_puncts=True):
    sys.path.append(os.path.join(os.environ['HOME'], 'codes', 'eval_cap'))
    from bleu.bleu import Bleu
    from meteor.meteor import Meteor
    from rouge.rouge import Rouge
    from cider.cider import Cider
    from spice.spice import Spice

    PUNCTUATIONS = set(["''", "'", "``", "`", "(", ")", "/", '"', 
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"])

    def clean_punctuation_in_sentence(sent):
        tokens = sent.strip().split(' ')
        tokens = [w for w in tokens if len(w) > 0 and w not in PUNCTUATIONS]
        sent = ' '.join(tokens)
        return sent
 
    preds = {}
    for key, value in pred_caps.items():
        if remove_puncts:
            preds[key] = [clean_punctuation_in_sentence(value[0].lower())]
        else:
            preds[key] = value

    refs = {}
    for key in preds.keys():
        if remove_puncts:
            refs[key] = [clean_punctuation_in_sentence(sent.lower()) for sent in ref_caps[key]]
        else:
            refs[key] = ref_caps[key]

    scorers = {
        'bleu4': Bleu(4),
        'meteor': Meteor(),
        'rouge': Rouge(),
        'cider': Cider(),
        'spice': Spice(),
    }
    if scorer_names is None:
        scorer_names = list(scorers.keys())

    scores = {}
    for measure_name in scorer_names:
        scorer = scorers[measure_name]
        s, _ = scorer.compute_score(refs, preds)
        if measure_name == 'bleu4':
            scores[measure_name] = s[-1] * 100
        else:
            scores[measure_name] = s * 100

    scorers['meteor'].meteor_p.kill()
    unique_words = set()
    sent_lens = []
    for key, value in preds.items():
        for sent in value:
            unique_words.update(sent.split())
            sent_lens.append(len(sent.split()))
    scores['num_words'] = len(unique_words)
    scores['avg_lens'] = np.mean(sent_lens)
    return scores