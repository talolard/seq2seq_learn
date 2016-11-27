from collections import namedtuple

import tensorflow as tf
from tensorflow.python.ops import rnn_cell, seq2seq
from tensorflow.python.ops.rnn import bidirectional_rnn
import tensorflow.contrib.rnn
from tensorflow.python.ops.seq2seq import sequence_loss_by_example


def LSTM_factory(args,factor=1):
    cell = rnn_cell.BasicLSTMCell(num_units=args.hidden_size*factor,
                                  state_is_tuple=True,
                                  activation=tf.tanh
                                  )
    stacked_cells = rnn_cell.MultiRNNCell(cells=[cell]*args.num_layers) #TODO change 3 to arg.num_layers
    return stacked_cells

def encoder(args,inputs=None):

    with tf.variable_scope("Model"):
        if inputs is None:
            inputs = tf.placeholder(tf.int32,shape=[args.batch_size,args.sequence_length])
        with tf.variable_scope("Embedding"):
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding4', [args.vocab_size, args.hidden_size])
                embedded = tf.nn.embedding_lookup(embedding, inputs)
                inputs = tf.unpack(embedded, axis=1)
        with tf.variable_scope("RNN_layer"):
            stacked_cells_fwd = LSTM_factory(args)
            stacked_cells_bwd = LSTM_factory(args)
            outputs, output_states_fw, output_states_bw =bidirectional_rnn(cell_fw=stacked_cells_fwd,
                              cell_bw=stacked_cells_bwd,
                              inputs=inputs,
                             dtype=tf.float32

                              )
            encoder_states = prepare_encoder_states(output_states_bw, output_states_fw)
            attn_size = stacked_cells_fwd.output_size +stacked_cells_bwd.output_size

            attn_states = tf.concat(1, [tf.reshape(e, [-1, 1, attn_size])
                                        for e in outputs])

            return outputs, attn_states,encoder_states


def prepare_encoder_states(output_states_bw, output_states_fw):
    '''
    Our encoder gives back lists of LSTMStateTuple for both the forward and backward rnns.
    The thing is, we need to combine them so that we have batch_size states, not 2*batch_size states
    :param output_states_bw: The output states of the backwards rnn
    :param output_states_fw: The output states of the forwards rnn
    :return: a list of length batch_size LSTMStateTuple with hidden states 2*len(output_states.hidden_state)
    '''
    encoder_states = []
    for output_state_fw, output_state_bw in zip(output_states_fw, output_states_bw):
        new_c = tf.concat(1, [output_state_fw.c, output_state_bw.c])
        new_h = tf.concat(1, [output_state_fw.h, output_state_bw.h])
        encoder_states.append(rnn_cell.LSTMStateTuple(new_c, new_h))
    return encoder_states


def decoder(args,targets,encoder_state, attn_states):
    decoder_cell =LSTM_factory(args,2)
    outputs, states = seq2seq.attention_decoder(
        decoder_inputs=targets,
        initial_state=encoder_state,
        attention_states=attn_states,
        cell=decoder_cell
    )
    return outputs,states

def interperter(outputs,targets):
    loss = sequence_loss_by_example(
        outputs[-1], targets)
    return loss



def test_encoder():
    args = namedtuple('args', ['vocab_size', 'hidden_size', 'sequence_length', 'batch_size', 'dropout','num_layers'])
    args.sequence_length=10
    args.vocab_size=10
    args.hidden_size=10
    args.batch_size=10
    args.dropout=0.9
    args.num_layers=1
    encoder(args)



def test_decoder():
        args = namedtuple('args', ['vocab_size', 'hidden_size', 'sequence_length', 'batch_size', 'dropout', 'num_layers'])
        args.sequence_length = 2
        args.vocab_size = 10
        args.hidden_size = 3
        args.batch_size = 5
        args.dropout = 0.9
        args.num_layers = 1

        dec_inp = [tf.constant(1, shape=[args.batch_size,args.sequence_length])]
        outputs, attn_states, encoder_state = encoder(args,dec_inp)
        output,states = decoder(args,outputs,encoder_state=encoder_state,attn_states=attn_states)
        #sess.run([output],)


test_decoder()