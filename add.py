import numpy as np
import tensorflow as tf
from new_npi import NPI, PRGENV, MovePTR, Write, Program

def C(M, k, beta = 100):
    M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), 1, keep_dims=True)) # N * 1
    k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), 1, keep_dims=True))
    M_dot_k = tf.matmul(M, tf.reshape(k, [-1, 1])) # N * 1

    similarity = M_dot_k * beta / (M_norm * k_norm + 1e-3)
    return tf.transpose(similarity) # 1 * N

def D(M, k, beta = 100):
    similarity = -beta * tf.square(M - k)
    return tf.nn.softmax(tf.transpose(similarity))

class Add1(Program):
    
    def do(self, env, args):
        env.add(args)
        return True

class Carry1(Program):

    def do(self, env, args):
        env.carry(args)
        return True
    
class AddEnv(PRGENV):
    def __init__(self, max_seq_length = 2, max_key_size = 4, max_arg_size = 3, max_env_size = 4,
                       lstm_layers = 2, lstm_hidden_size = 256, mlp_layers = 1, mlp_hidden_size = 128,
                       enc_out_size = 128, end_out_size = 1, prg_out_size = 4, arg_out_size = 10, env_out_size = 10):
        super(AddEnv, self).__init__(max_seq_length, max_key_size, max_arg_size, max_env_size, 
                                     lstm_layers, lstm_hidden_size, mlp_layers, mlp_hidden_size,
                                     enc_out_size, end_out_size, prg_out_size, arg_out_size, env_out_size)
        self.prg_keys = tf.eye(max_key_size, prg_out_size)
        self.prgs = [MovePTR('MOVE_PTR'), Add1('ADD1'), Carry1('CARRY1'), Program('ADD', self.gen_weights())]
        self.envs = tf.placeholder(tf.float32, [self.max_env_size, self.max_seq_length, self.env_out_size], name='input_env')
        self.ptrs = weight('ptr_init_b', [1, self.max_env_size], init='zero')
        self.ADDR = tf.reshape(tf.to_float(tf.range(self.max_seq_length)), [-1, 1])

    def get_prg(self, K):
        prg_id = tf.argmax(C(self.prg_keys, K), 1)
        case_dict = { tf.equal(prg_id, i): self.prgs[i].info for i in range(self.max_key_size) }
        return tf.case(case_dict, default=None, exclusive=True)

    def run(self, K, Args):
        prg_id = tf.argmax(C(self.prg_keys, K), 1)
        case_dict = { tf.equal(prg_id, i): self.prgs[i].do(self, Args) for i in range(self.max_key_size) }
        return tf.case(case_dict, default=None, exclusive=True)

    def get_env(self):
        a_r = D(tf.tile(self.ADDR, [1, self.max_env_size]), self.ptrs) # max_env_size * max_seq_length
        val = tf.reshape(tf.batch_matmul(tf.expand_dims(a_r, 1), self.envs), [self.max_env_size, self.env_out_size])
        return val

    def write(self, Args):
        row = tf.transpose(tf.softmax(Args[0][:, 0:2])) # 2 * 1
        value = tf.tile(Args[1], [2, 1])
        a_w = D(tf.tile(self.ADDR, [1, 2]), self.ptrs[:, -2:]) * row # 2 * max_seq_length
        out = self.envs[-2:, :, :] * tf.expand_dims(1.0 - a_w, 2) + tf.expand_dims(value, 1) * tf.expand_dims(a_w, 2)
        self.envs = tf.concat(0, [self.envs[0:self.max_env_size-2,:,:], out])
    
    def moveptr(self, Args):
        row = tf.nn.softmax(Args[0][:, 0:self.max_env_size])
        delta = tf.argmax(tf.nn.softmax(Args[1]), 1)
        self.ptrs = self.ptrs + delta * row
        
    def add(self, Args):
        pass

    def carry(self, Args):
        pass