from collections import namedtuple
from math import sqrt

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.ops import array_ops
from tensorflow.contrib.rnn.python.ops.rnn_cell import AttentionCellWrapper
from tensorflow.python.ops import rnn_cell, seq2seq
from tensorflow.python.ops.rnn import bidirectional_rnn
import tensorflow.contrib.rnn
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops.seq2seq import sequence_loss_by_example
from tensorflow.python.training.adam import AdamOptimizer


def LSTM_factory(hidden_size,num_layers,dropout):
    DropoutWrapper
    cell = rnn_cell.LSTMCell(num_units=hidden_size,
                                  state_is_tuple=True,
                                  activation=tf.tanh,
                                initializer=tf.contrib.layers.xavier_initializer()
                                  )
    cell =DropoutWrapper(cell=cell,input_keep_prob=dropout,output_keep_prob=dropout)
    stacked_cells = rnn_cell.MultiRNNCell(cells=[cell]*num_layers) #TODO change 3 to arg.num_layers
    return stacked_cells

class Model():
    def __init__(self,args):
        self.inputs  = tf.placeholder(tf.int32, shape=[args.batch_size, args.sequence_length])
        self.targets = tf.placeholder(tf.int32, shape=[args.batch_size, args.sequence_length])
        with tf.name_scope("embedding"):
            embedding_size = int(sqrt(args.vocab_source_size)+1)
            embedding = tf.get_variable('embedding',
                                        shape= [args.vocab_source_size, embedding_size],#embed them in a small space
                                        initializer=tf.contrib.layers.xavier_initializer()
                                        )
            embedded = tf.nn.embedding_lookup(embedding, self.inputs)
            embedded_inputs = tf.unpack(embedded, axis=0)

        with tf.variable_scope("bidi_rnn") as bidi_scope:
            cell = LSTM_factory(args.hidden_size,args.num_layers,dropout=args.dropout)
            outputs,fwd_state,bwd_state= tf.nn.bidirectional_rnn(cell_fw=cell,
                                                            cell_bw=cell,
                                                            inputs=embedded_inputs,
                                                            dtype=tf.float32)

        with tf.variable_scope("decoder_rnn"):
             decoder_cell = LSTM_factory(args.hidden_size, args.num_layers*2,dropout=args.dropout)
             decoder_cell = AttentionCellWrapper(cell=decoder_cell,attn_length=args.hidden_size,state_is_tuple=True)
             final_outputs,state = tf.nn.rnn(cell=decoder_cell,inputs=outputs,dtype=tf.float32)

        with tf.variable_scope("logits") as logits_scope:
            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(final_outputs, [-1, args.hidden_size])

            logits = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=args.vocab_target_size,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                scope=logits_scope)
            logits = tf.reshape(logits, [args.batch_size, -1, args.vocab_target_size])

            self.logits =logits

        with tf.variable_scope("loss"):
            flat_targets = tf.reshape(self.targets, [-1])
            flat_logits = tf.reshape(logits, [-1, args.vocab_target_size])
            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(flat_logits, flat_targets)

            batch_loss = tf.reduce_sum(losses,name="batch_loss")
            tf.contrib.losses.add_loss(batch_loss)
            total_loss = tf.contrib.losses.get_total_loss()

            # Add summaries.
            tf.scalar_summary("batch_loss", batch_loss)
            tf.scalar_summary("total_loss", total_loss)


            self.total_loss = total_loss
            self.batch_loss = batch_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.

        with tf.name_scope("optimization"):
            opt = AdamOptimizer(learning_rate=args.learning_rate)
            gvs = opt.compute_gradients(self.batch_loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = opt.apply_gradients(capped_gvs)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)



        for grad, var in gvs:

            if grad is not None:
                print(capped_gvs)
                tf.histogram_summary(var.op.name + '/gradients', grad,)

        with tf.name_scope("tensors"):
            self.train_op = train_op
            self.logits = logits
            self.total_loss = total_loss
            self.summaries =tf.merge_all_summaries()




