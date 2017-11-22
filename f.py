import tensorflow as tf

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class f(object):
    def __init__(self, input_size, action_num):
        self.grad_clip = 5.0
        self.input_size = input_size
        self.W1 = tf.get_variable('W1', [self.input_size*6 + 6, 2* self.input_size])
        self.b1 = tf.get_variable('b1', [ 2* self.input_size])
        self.W2 = tf.get_variable('W2', [2*self.input_size, int(1.5 * self.input_size)])
        self.b2 = tf.get_variable('b2', [ int(1.5 * self.input_size)])
        self.W3 = tf.get_variable('W3', [ int(1.5 * self.input_size), action_num])
        self.b3 = tf.get_variable('b3', [action_num])
        self.W4 = tf.get_variable('w4', [ int(1.5 * self.input_size),1])

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
        self.P = tf.nn.softmax(tf.nn.xw_plus_b(layer3, self.W3, self.b3))
        self.V = tf.matmul(layer3, self.W4)


        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.pi = tf.placeholder('float32', [None, action_num], name='pi')

        # action_one_hot = tf.one_hot(self.action, action_num, 1.0, 0.0, name='action_one_hot')
        # q_acted = tf.reduce_sum(self.V * action_one_hot, reduction_indices=1, name='q_acted')
        self.delta = self.target_q_t - self.V[:,0]

        time = tf.constant(0, dtype=tf.int32)
        batch_size = tf.shape(self.pi)[0]

        inputs_pi = tf.TensorArray(dtype=tf.float32, size=batch_size)
        inputs_pi = inputs_pi.unstack(self.pi)

        inputs_P = tf.TensorArray(dtype=tf.float32, size=batch_size)
        inputs_P = inputs_P.unstack(self.P)
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        f0 = tf.zeros([1], dtype=tf.bool)

        def loop_fn(t, pi_t, P_t, emit_ta, finished):
            o_t = tf.matmul(tf.expand_dims(pi_t, 0 ), tf.expand_dims(P_t, 1))[0,0]
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t + 1, [batch_size])
            pi_nt =  tf.cond(tf.reduce_all(finished), lambda: tf.zeros([action_num], dtype=tf.float32), lambda: inputs_pi.read(t+1))
            P_nt =  tf.cond(tf.reduce_all(finished), lambda: tf.zeros([action_num], dtype=tf.float32), lambda: inputs_P.read(t+1))
            return t + 1, pi_nt, P_nt, emit_ta, finished

        _, _, _, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_pi.read(0), inputs_P.read(0), emit_ta, f0))
        mul = emit_ta.stack()
        self.tt= tf.shape(mul), tf.shape(self.pi), tf.shape(self.P)
        self.loss = self.delta ** 2 - mul


        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def __call__(self, state, sess):
        # print ('dsf', state['ptr_result'].shape, tf.get_variable_scope().name)
        # print tf.trainable_variables()
        values = sess.run([self.P, self.V], feed_dict={
            self.inputA : state['inputA'][None,:],
            self.inputB : state['inputB'][None,:],
            self.inputcarry : state['inputcarry'][None,:],
            self.input_semi_result : state['input_semi_result'][None,:],
            self.ptr_carry : state['ptr_carry'][None,:],
            self.ptr_result : state['ptr_result'][None,:]
        })
        # print ('end')
        return values

    def predict(self, state, sess):
        print ('predict')
        return self.__call__(state, sess)

    def train(self,session, x):
        # print ('train')
        session.run(self.train_op, feed_dict = {
            self.inputA: x['inputA'],
            self.inputB: x['inputB'],
            self.inputcarry: x['inputcarry'],
            self.input_semi_result: x['input_semi_result'],
            self.ptr_carry: x['ptr_carry'],
            self.ptr_result: x['ptr_result'],
            self.target_q_t: x['target_q_t'],
            self.pi: x['pi']
        })

    def debug(self, session, x):
        # print ('dsf', state['ptr_result'].shape, tf.get_variable_scope().name)
        # print tf.trainable_variables()
        values = session.run(self.tt, feed_dict = {
            self.inputA: x['inputA'],
            self.inputB: x['inputB'],
            self.inputcarry: x['inputcarry'],
            self.input_semi_result: x['input_semi_result'],
            self.ptr_carry: x['ptr_carry'],
            self.ptr_result: x['ptr_result'],
            self.target_q_t: x['target_q_t'],
            self.pi: x['pi']
        })
        print values
