import argparse
import os
import pickle

import numpy as np
import codecs
import collections
import re
MAX_SIZE=100
PAD='ה'
punctation_regex=re.compile('''[.,?!;']''')
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str,
                    default='data/wikipedia/input.txt', help="data file in utf-8")
parser.add_argument('--data_dir', type=str,
                    default='data/wikipedia', help="save directory for preprocessed files")
parser.add_argument('--val_frac', type=float,
                    default=0.1, help="fraction of data to use as validation set")
parser.add_argument('--test_frac', type=float,
                    default=0.1, help="fraction of data to use as test set")
parser.add_argument('--verbose', action='store_true',
                    help="verbose printing")
args = parser.parse_args()

def prepare_dataset(args):
    vocab_file_source = os.path.join(args.data_dir, "vocab_source.pkl")
    train_file_source = os.path.join(args.data_dir, "train_source.npy")
    val_file_source = os.path.join(args.data_dir, "val_source.npy")
    test_file_source = os.path.join(args.data_dir, "test_source.npy")
    
    vocab_file_target = os.path.join(args.data_dir, "vocab_target.pkl")
    train_file_target = os.path.join(args.data_dir, "train_target.npy")
    val_file_target = os.path.join(args.data_dir, "val_target.npy")
    test_file_target = os.path.join(args.data_dir, "test_target.npy")


    with open(args.input_file,'r',encoding='utf-8') as f:
        target = f.readlines()
        target = list(map(lambda x:x.strip()[:MAX_SIZE].ljust(MAX_SIZE,PAD),target))

    source = list(map(lambda x:re.sub(punctation_regex,'',x).ljust(MAX_SIZE,PAD),target)) #Strip away punctuation


    
    
    target_vocab = make_vocabulary(target, vocab_file_target)
    make_data_tensors(args, target_vocab, target, test_file_target, train_file_target, val_file_target)
    source_vocab = make_vocabulary(source, vocab_file_source)
    make_data_tensors(args, source_vocab, source, test_file_source, train_file_source, val_file_source)
    print('source_vocab {0} target_vocab {1}'.format(len(source_vocab),len(target_vocab)))



def make_data_tensors(args, vocab, txt, test_file, train_file, val_file):
    id_lines=[]
    for line in txt:
        id_lines.append(np.array(list(map(vocab.get,line))))
    data_tensor = np.array(id_lines)
    data_size = data_tensor.shape[0]
    val_size = int(args.val_frac * data_size)
    test_size = int(args.test_frac * data_size)
    train_size = data_size - val_size - test_size
    np.save(train_file, data_tensor[:train_size])
    np.save(val_file, data_tensor[train_size:train_size + val_size])
    np.save(test_file, data_tensor[train_size + val_size:data_size])


def make_vocabulary(txt, vocab_file):
    txt =''.join(txt)
    counter = collections.Counter(txt)
    counts = sorted(counter.items(), key=lambda x: -x[1])
    tokens, _ = zip(*counts)
    vocab_size = len(tokens)
    vocab = dict(zip(tokens, range(vocab_size)))
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab


prepare_dataset(args)
