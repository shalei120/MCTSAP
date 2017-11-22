import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class NPICTL(object):
    def __init__(self, mlp_layers = 1, lstm_layers = 2):
        self.mlp_layers = mlp_layers
        self.lstm_layers = lstm_layers

    def __call__(self, Env, Args, Program, W_list, B_list, output_list, hidden_list):
        S = self.MLP(tf.concat(1, [Env] + Args), W_list[1], B_list[1], "enc")
        new_output_list, new_hidden_list = self.LSTM(tf.concat(1, [S, Program]), output_list, hidden_list, W_list[0], B_list[0], "lstm")
        R = tf.nn.sigmoid(self.MLP(new_output_list[-1], W_list[2], B_list[2], "end"))
        K = self.MLP(self.MLP(new_output_list[-1], W_list[3], B_list[3], "prg"))
        new_args = []
        for i in range(len(Args)):
            new_args.append(self.MLP(new_output_list[-1], W_list[4+i], B_list[4+i], "arg%d"%i))
        return new_args, K, R, new_output_list, new_hidden_list

    def MLP(self, X, W, B, scope):
        with tf.variable_scope("ctl_" + scope):
            for layer_idx in range(self.mlp_layers):
                X = tf.nn.relu(tf.add(tf.matmul(X, W[layer_idx]), B[layer_idx]))
            return X

    def LSTM(self, input_, output_list_prev, hidden_list_prev, W, B, scope):
        """Build LSTM controller."""

        with tf.variable_scope("ctl_" + scope):
            output_list = []
            hidden_list = []
            input_size = input_.get_shape().as_list()[1]
            hidden_size = (W.get_shape().as_list()[0]/4 - input_size) / (2*self.lstm_layers - 1)
            first_layer_size = 4*(input_size+hidden_size)
            W_layers = [W[0:first_layer_size, :]] + tf.split(0, self.lstm_layers-1, W[first_layer_size:, :])
            B_layers = tf.split(1, self.lstm_layers, B, 'B_layer_split')
            for layer_idx in range(self.lstm_layers):
                o_prev = output_list_prev[layer_idx]
                h_prev = hidden_list_prev[layer_idx]
                W_gates = tf.split(0, 4, W_layers[layer_idx], 'W_gate_split')
                B_gates = tf.split(1, 4, B_layers[layer_idx], 'B_gate_split')

                if layer_idx == 0:
                    def new_gate(gate_id):
                        return math_ops.matmul(array_ops.concat(1, [input_, o_prev]), W_gates[gate_id]) + B_gates[gate_id]
                else:
                    def new_gate(gate_id):
                        return math_ops.matmul(array_ops.concat(1, [output_list[-1], o_prev]), W_gates[gate_id]) + B_gates[gate_id]

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate(0))
                f = tf.sigmoid(new_gate(1))
                o = tf.sigmoid(new_gate(2))
                update = tf.tanh(new_gate(3))

                # update the sate of the LSTM cell
                hid = tf.add_n([f * h_prev, i * update])
                out = o * tf.tanh(hid)

                hidden_list.append(hid)
                output_list.append(out)

            return output_list, hidden_list