from __future__ import print_function
import os
import numpy as np
from six.moves import cPickle as pickle


class DataLoader():

    def __init__(self, args):
        self.args = args

        vocab_target_file = os.path.join(self.args.data_dir, "vocab_target.pkl")
        vocab_source_file = os.path.join(self.args.data_dir, "vocab_source.pkl")
        train_source_file = os.path.join(args.data_dir, "train_source.npy")
        val_source_file = os.path.join(args.data_dir, "val_source.npy")
        test_source_file = os.path.join(args.data_dir, "test_source.npy")
        
        train_target_file = os.path.join(args.data_dir, "train_target.npy")
        val_target_file = os.path.join(args.data_dir, "val_target.npy")
        test_target_file = os.path.join(args.data_dir, "test_target.npy")

        
        assert os.path.exists(args.data_dir), "data directory does not exist"
        assert os.path.exists(vocab_target_file), "vocab file does not exist"
        assert os.path.exists(vocab_source_file), "vocab file does not exist"
        assert os.path.exists(train_target_file), "train file does not exist"
        assert os.path.exists(val_target_file), "validation file does not exist"
        assert os.path.exists(test_target_file), "test file does not exist"
        assert os.path.exists(train_source_file), "train file does not exist"
        assert os.path.exists(val_source_file), "validation file does not exist"
        assert os.path.exists(test_source_file), "test file does not exist"


        with open(vocab_target_file, 'rb') as f:
            self.vocab_target = pickle.load(f)
        with open(vocab_source_file, 'rb') as f:
            self.vocab_source = pickle.load(f)

        self.vocab_source_size = len(self.vocab_source)
        self.vocab_target_size = len(self.vocab_target)

        sets = {}

        for label, source_file,target_file in [("train", train_source_file, train_target_file), ("validation", val_source_file,val_target_file), ("test", test_source_file,test_target_file)]:
            source_tensor = np.load(source_file)
            target_tensor = np.load(target_file)




            x = source_tensor[:round(source_tensor.shape[0] -100,-2)]
            y = target_tensor[:round(target_tensor.shape[0] -100,-2)]
            num_batches = int(x.shape[0] / (args.batch_size))
            x_batches = np.split(x, num_batches, )
            y_batches = np.split(y, num_batches, )
            self.sequence_length = x.shape[1]
            sets[label] = BatchIterator(x_batches, y_batches)

            if args.verbose:
                print("{} data loaded".format(label))
                print("number of batches: {}".format(num_batches))

        self.train = sets["train"]
        self.val = sets["validation"]
        self.test = sets["test"]


class BatchIterator():

    def __init__(self, x_batches, y_batches):
        self.x_batches = x_batches
        self.y_batches = y_batches
        self.num_batches = len(x_batches)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.num_batches:
            raise StopIteration
        else:
            x, y = self.x_batches[self.counter], self.y_batches[self.counter] 
            self.counter += 1
            return x, y

    next = __next__ # python 2
