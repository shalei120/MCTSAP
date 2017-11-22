import tensorflow as tf

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class Q(object):
    def __init__(self, input_size, action_num):
        self.grad_clip = 5.0
        self.input_size = input_size
        self.W1 = tf.get_variable('W1', [self.input_size*6 + 6, 2* self.input_size])
        self.b1 = tf.get_variable('b1', [ 2* self.input_size])
        self.W2 = tf.get_variable('W2', [2*self.input_size, int(1.5 * self.input_size)])
        self.b2 = tf.get_variable('b2', [ int(1.5 * self.input_size)])
        self.W3 = tf.get_variable('W3', [ int(1.5 * self.input_size), action_num])
        self.b3 = tf.get_variable('b3', [action_num])

        self.inputA = tf.placeholder(tf.int32, [None, self.input_size+1])
        self.inputB = tf.placeholder(tf.int32, [None, self.input_size+1])
        self.inputcarry = tf.placeholder(tf.int32, [None, self.input_size+1])
        self.input_semi_result = tf.placeholder(tf.int32, [None, self.input_size+1])
        self.ptr_carry = tf.placeholder(tf.int32, [None, self.input_size+1])   # 0000100000
        self.ptr_result = tf.placeholder(tf.int32, [None, self.input_size+1])

        layer1 = tf.concat([self.inputA, self.inputB, self.inputcarry, self.input_semi_result, self.ptr_carry, self.ptr_result], axis = 1)
        layer1 = tf.cast(layer1, tf.float32)
        layer2 = tf.tanh(tf.nn.xw_plus_b(layer1, self.W1, self.b1))
        layer3 = tf.tanh(tf.nn.xw_plus_b(layer2, self.W2, self.b2))
        self.value = tf.nn.softmax(tf.nn.xw_plus_b(layer3, self.W3, self.b3))

        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.action = tf.placeholder('int32', [None], name='action')

        action_one_hot = tf.one_hot(self.action, action_num, 1.0, 0.0, name='action_one_hot')
        q_acted = tf.reduce_sum(self.value * action_one_hot, reduction_indices=1, name='q_acted')
        self.delta = self.target_q_t - q_acted
        self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def __call__(self, state, sess):
        # print ('dsf', state['ptr_result'].shape, tf.get_variable_scope().name)
        # print tf.trainable_variables()
        values = sess.run(self.value, feed_dict={
            self.inputA : state['inputA'],
            self.inputB : state['inputB'],
            self.inputcarry : state['inputcarry'],
            self.input_semi_result : state['input_semi_result'],
            self.ptr_carry : state['ptr_carry'],
            self.ptr_result : state['ptr_result']
        })
        # print ('end')
        return values

    def predict(self, state, sess):
        print ('predict')
        return self.__call__(state, sess)

    def train(self,session, x):
        print ('train')
        session.run(self.train_op, feed_dict = {
            self.inputA: x['inputA'],
            self.inputB: x['inputB'],
            self.inputcarry: x['inputcarry'],
            self.input_semi_result: x['input_semi_result'],
            self.ptr_carry: x['ptr_carry'],
            self.ptr_result: x['ptr_result'],
            self.target_q_t: x['target_q_t'],
            self.action: x['action']
        })
