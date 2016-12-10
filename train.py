from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import os
import time
from six.moves import cPickle as pickle

from utils import DataLoader
from model import punctuator


def train(args):
    if args.verbose:
        print(vars(args))

    loader = DataLoader(args)

    if args.init_from is not None:
        assert os.path.exists(args.init_from),"{} is not a file or directory".format(args.init_from)

        if os.path.isdir(args.init_from):
            parent_dir = args.init_from
        else:
            parent_dir = os.path.dirname(args.init_from)

        config_file = os.path.join(parent_dir, "config.pkl")
        vocab_file = os.path.join(parent_dir, "vocab.pkl")

        assert os.path.isfile(config_file), "config.pkl does not exist in directory {}".format(parent_dir)
        assert os.path.isfile(vocab_file), "vocab.pkl does not exist in directory {}".format(parent_dir)

        if os.path.isdir(args.init_from):
            checkpoint = tf.train.latest_checkpoint(parent_dir)
            assert checkpoint, "no checkpoint in directory {}".format(args.init_from)
        else:
            checkpoint = args.init_from

        with open(os.path.join(parent_dir, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)

        assert saved_args.hidden_size == args.hidden_size, "hidden size argument ({}) differs from save ({})" \
                                                            .format(saved_args.hidden_size, args.hidden_size)
        assert saved_args.num_layers == args.num_layers, "number of layers argument ({}) differs from save ({})" \
                                                            .format(saved_args.num_layers, args.num_layers)

        with open(os.path.join(parent_dir, 'vocab.pkl'), 'rb') as f:
            saved_vocab = pickle.load(f)

        assert saved_vocab == loader.vocab, "vocab in data directory differs from save"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    new_config_file = os.path.join(args.save_dir, 'config.pkl')
    new_vocab_file = os.path.join(args.save_dir, 'vocab.pkl')

    if not os.path.exists(new_config_file):
        with open(new_config_file, 'wb') as f:
            pickle.dump(args, f)
    args.sequence_length = loader.sequence_length
    args.vocab_source_size =loader.vocab_source_size
    args.vocab_target_size  = loader.vocab_target_size

    model = punctuator.Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        summary_writer = tf.train.SummaryWriter('./logs/7', graph=tf.get_default_graph())
        val_summary_writer = tf.train.SummaryWriter('./logs/7/val', graph=tf.get_default_graph())

        saver = tf.train.Saver(tf.all_variables())

        if args.init_from is not None:
            try:
                saver.restore(sess, checkpoint)
            except ValueError:
                print("{} is not a valid checkpoint".format(checkpoint))

            if args.verbose:
                print("initializing from {}".format(checkpoint))

        for e in range(args.num_epochs):
            for b, (x, y) in enumerate(loader.train):
                global_step = e * loader.train.num_batches + b
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        #model.dropout: args.dropout, TODO add dropout
                        }
                train_loss, train, summaries= sess.run([model.total_loss, model.train_op,model.summaries], feed)
                summary_writer.add_summary(summaries, global_step=global_step)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(global_step,
                            args.num_epochs * loader.train.num_batches,
                            e, train_loss, end - start))

                if  (global_step >1) and (global_step % args.save_every == 0 \
                    or (e == args.num_epochs - 1 and b == loader.train.num_batches - 1)):
                    all_loss = 0


                    checkpoint_path = os.path.join(args.save_dir, 'iter_{}-.ckpt' \
                                        .format(global_step, ))
                    saver.save(sess, checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
        val_loss = run_validation(all_loss, global_step, loader, model, sess, start, val_summary_writer)


def run_validation(all_loss, global_step, loader, model, sess, start, val_summary_writer):
    for b, (x, y) in enumerate(loader.val):
        feed = {model.inputs: x,
                model.targets: y}
        val_loss,  val_summaries = sess.run([model.total_loss,  model.summaries],
                                                feed)
        val_summary_writer.add_summary(val_summaries, global_step=global_step + b)
        all_loss += val_loss
    end = time.time()
    val_loss = all_loss / loader.val.num_batches
    print("val_loss = {:.3f}, time/val = {:.3f}".format(val_loss, end - start))
    return val_loss


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/wikipedia', help="directory with processed data")
    parser.add_argument('--init_from', type=str,
                        default=None, help="checkpoint file or directory to intialize from, if directory the most recent checkpoint is used")
    parser.add_argument('--save_every', type=int,
                        default=1024, help="batches per save, also reports loss on validation set")
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints', help="directory to save checkpoints and config files")
    parser.add_argument('--num_epochs', type=int,
                        default=128, help="number of epochs to train")
    parser.add_argument('--batch_size', type=int,
                        default=10  , help="minibatch size")
    parser.add_argument('--vocab_size', type=int,
                        default=None, help="vocabulary size, defaults to infer from the input")
    parser.add_argument('--seq_length', type=int,
                        default=64, help="sequence length")
    parser.add_argument('--learning_rate', type=float,
                        default=0.0008, help="learning rate")
    parser.add_argument('--decay_factor', type=float,
                        default=0.97, help="learning rate decay factor")
    parser.add_argument('--decay_every', type=int,
                        default=1, help="how many epochs between every application of decay factor")
    parser.add_argument('--grad_clip', type=float,
                        default=5, help="maximum value for gradients, set to 0 to remove gradient clipping")
    parser.add_argument('--hidden_size', type=int,
                        default=200, help="size of hidden units in network")
    parser.add_argument('--num_layers', type=int,
                        default=2, help="number of hidden layers in network")
    parser.add_argument('--dropout', type=float,
                        default=0.7, help="dropout keep probability applied to input between lstm layers")
    parser.add_argument('--verbose', action='store_true',
                        help="verbose printing")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_train_args()
    train(args)
