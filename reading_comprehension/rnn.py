"""
Goal:
    - Create RNN layers

Important Concepts/Design Choices:
    - For "rnn" it is difficult in TF to iterate over variable number of iterations based
        on the value of a tensor (I couldn't figure this out, nor could I find any
        examples of others doing this). Thus how is it possible to iterate for variable
        number of time steps for each batch? This is where the use of TF's control
        flow options come in, namely tf.while_loop. See inline comments for details.

TODO/FIX: Get iteration numbers for RNN? Scope

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

import tensorflow as tf

def rnn(cell, inputs, seq_lens_mask, start_state=None):
    batch_size = tf.shape(inputs)[0]
    embedding_dim = inputs.get_shape().as_list()[2]

    if start_state == None:
        state = cell.zero_state(batch_size)
    else:
        state = start_state

    # Find the maximum document length, set as total number of time steps
    time = tf.cast(tf.reduce_max(tf.reduce_sum(seq_lens_mask, 1)), tf.int32)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs):
        # FIGURE OUT HOW TO GET PROPER NUMBERS HERE: with tf.variable_scope("Cell-Time{}".format(1)):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_lens_mask, [0, i], [batch_size, 1])
            # Squeeze to get correct dimensions - dim 1 goes to 0
            input_ = tf.squeeze(input_, [1])

            # RNN time step
            output, state = cell(input_, state, time_mask)
            output = tf.expand_dims(output, 1)

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, state, outputs]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    outputs_shape = tf.TensorShape([None, None, cell._state_size])

    # Run RNN while loop
    _, _, last_state, hidden_states = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, state, tf.zeros([batch_size, 1, cell._state_size])], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), state.get_shape(), outputs_shape])

    # get rid of zero output start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1, 0], [batch_size, -1, cell._state_size])
    # reshape hidden_states to (batch x time x hidden_state_size)
    #hidden_states = tf.reshape(hidden_states, [batch_size, -1, cell._state_size])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state)


def bidirectional_rnn(forward_cell, backward_cell, inputs, seq_lens_mask, concatenate=True):
    seq_lens = tf.cast(tf.reduce_sum(seq_lens_mask, 1), tf.int32)

    # Reverse inputs (batch x time x embedding_dim); takes care of variable seq_len
    reverse_inputs = tf.reverse_sequence(inputs, seq_lens, seq_dim=1, batch_dim=0)

    # Run forwards and backwards RNN
    forward_outputs, forward_last_state = \
        rnn(forward_cell, inputs, seq_lens_mask)
    backward_outputs_reversed, backward_last_state = \
        rnn(backward_cell, reverse_inputs, seq_lens_mask)

    backward_outputs = tf.reverse_sequence(backward_outputs_reversed, seq_lens, seq_dim=1, batch_dim=0)

    if concatenate:
        # last_state dimensions: batch x hidden_size
        last_state = tf.concat(1, [forward_last_state, backward_last_state])
        # outputs dimensions: batch x time x hidden_size
        outputs = tf.concat(2, [forward_outputs, backward_outputs])

        # Dimensions: outputs (batch x time x hidden_size*2); last_state (batch x hidden_size*2)
        return (outputs, last_state)

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (forward_outputs, forward_last_state, backward_outputs, backward_last_state)
