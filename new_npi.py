import numpy as np
import tensorflow as tf

from npi_controller import NPICTL
from utils import weight

class NPI(object):
    def __init__(self, prg_env, stop_threshold = 0.5):  
        self.prg_env = prg_env
        self.stop_threshold = stop_threshold
        self.ctl = NPICTL(self.prg_env.mlp_layers, self.prg_env.lstm_layers)

    def init_lstm_state(self):
        output_init_list = []
        hidden_init_list = []
        for idx in range(self.prg_env.lstm_layers):
            output_init_list.append(tf.zeros([1, self.prg_env.lstm_hidden_size]))
            hidden_init_list.append(tf.zeros([1, self.prg_env.lstm_hidden_size]))
        return output_init_list, hidden_init_list

    def run(self, Prg, Args):
        output_list, hidden_list = self.init_lstm_state()

        def bodyfunc(args, prg, output, hidden):
            prg_info = self.prg_env.get_prg(prg)
            args, prg, r, output, hidden = self.ctl(self.prg_env.get_env(), args, prg, prg_info[0], prg_info[1], output, hidden)
            if not self.prg_env.run(prg, args):
                self.run(prg, args)
            return args, prg, r, output, hidden

        condition = lambda args, prg, r, output, hidden: tf.less(r, self.stop_threshold)
        loopbody = lambda args, prg, r, output, hidden: bodyfunc(args, prg, output, hidden)
        tf.while_loop(condition, loopbody, loop_vars=[Args, Prg, tf.constant(0), output_list, hidden_list])

class PRGENV(object):
    def __init__(self, max_seq_length, max_key_size, max_arg_size, max_env_size,
                       lstm_layers, lstm_hidden_size, mlp_layers, mlp_hidden_size,
                       enc_out_size, end_out_size, prg_out_size, arg_out_size, env_out_size):
        self.max_seq_length = max_seq_length
        self.max_key_size = max_key_size
        self.max_arg_size = max_arg_size
        self.max_env_size = max_env_size
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.enc_out_size = enc_out_size
        self.end_out_size = end_out_size
        self.prg_out_size = prg_out_size
        self.arg_out_size = arg_out_size
        self.env_out_size = env_out_size

    def gen_weights(self):
        # init all weight and bias for a prg
        lstm_w_rows = (self.enc_out_size + self.prg_out_size + self.lstm_hidden_size) * 4 + self.lstm_hidden_size * 8 * (self.lstm_layers - 1)
        W_list = [weight('lstm_w', [lstm_w_rows, self.lstm_hidden_size], init='xavier'), [], [], []] + [] * self.max_arg_size
        B_list = [weight('lstm_b', [1, self.lstm_hidden_size * 4 * self.lstm_layers], init='contant'), [], [], []] + [] *self.max_arg_size

        for i in range(0, self.mlp_layers):
            if i == 0:
                enc_layer_in = self.max_env_size * self.env_out_size + self.max_arg_size * self.arg_out_size
                end_layer_in = prg_layer_in = arg_layer_in = self.lstm_hidden_size
            else:
                enc_layer_in = end_layer_in = prg_layer_in = arg_layer_in = self.mlp_hidden_size

            if i == self.mlp_layers-1:
                enc_layer_out, end_layer_out, prg_layer_out, arg_layer_out = self.enc_out_size, self.end_out_size, self.prg_out_size, self.arg_out_size
            else:
                enc_layer_out = end_layer_out = prg_layer_out = arg_layer_out = self.mlp_hidden_size

            W_list[1].append(weight('enc_l%d_w' % i, [enc_layer_in, enc_layer_out], init='xavier'))
            B_list[1].append(weight('enc_l%d_b' % i, [1, enc_layer_out], init='contant'))
            W_list[2].append(weight('end_l%d_w' % i, [end_layer_in, end_layer_out], init='xavier'))
            B_list[2].append(weight('end_l%d_b' % i, [1, end_layer_out], init='contant'))
            W_list[3].append(weight('prg_l%d_w' % i, [prg_layer_in, prg_layer_out], init='xavier'))
            B_list[3].append(weight('prg_l%d_b' % i, [1, prg_layer_out], init='conctant'))
            for j in range(len(self.max_arg_size)):
                W_list[4+i].append(weight('arg%d_l%d_w' % (j, i), [arg_layer_in, arg_layer_out], init='xavier'))
                B_list[4+i].append(weight('arg%d_l%d_b' % (j, i), [1, arg_layer_out], init='contant'))
        return [W_list, B_list]

    def get_env(self):
        raise NotImplementedError()

    def get_prg(self, K):
        raise NotImplementedError()

    def run(self, K, Args):
        raise NotImplementedError()

    def write(self, id, value):
        raise NotImplementedError()

    def moveptr(self, id, value):
        raise NotImplementedError()

class Program(object):

    def __init__(self, name, info = None):
        self.name = name
        self.program_id = None
        self.info = info

    def description(self, args):
        return "%s(%s)" % (self.name, ", ".join([str(x) for x in args]))

    def do(self, env, args):
        return False
        
class Write(Program):
    
    def do(self, env, args):
        env.write(args)
        return True

class MovePTR(Program):

    def do(self, env, args):
        env.moveptr(args)
        return True