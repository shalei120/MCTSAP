'''
author Lei Sha
'''

import random
import numpy as np
import tensorflow as tf
from npienv import *
from f import *
from mcst import *
from Memory import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """

    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers):
        with tf.variable_scope('plain'):
            self.model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)

        # print tf.trainable_variables()
        with tf.variable_scope('target'):
            self.targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)


        # print tf.trainable_variables()


    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else:
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform',
                                W_regularizer=l2(regularizationFactor), bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU"):
                model.add(LeakyReLU(alpha=0.01))
            else:
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU"):
                    model.add(LeakyReLU(alpha=0.01))
                else:
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        self.model = Q(inputs, outputs)
        return self.model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer ", i, ": ", weights)
            i += 1

    def backupNetwork(self, model, backup):
        print( 'backup')
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state, session):
        single_state = state.copy()
        single_state['inputA'] = single_state['inputA'][None,:]
        single_state['inputB'] = single_state['inputB'][None,:]
        single_state['input_semi_result'] = single_state['input_semi_result'][None,:]
        single_state['ptr_result'] = single_state['ptr_result'][None,:]
        single_state['inputcarry'] = single_state['inputcarry'][None,:]
        single_state['ptr_carry'] = single_state['ptr_carry'][None,:]
        predicted = self.model(single_state,session)
        return predicted

    def getTargetQValues(self, state):
        single_state = state.copy()
        single_state['inputA'] = single_state['inputA'][None,:]
        single_state['inputB'] = single_state['inputB'][None,:]
        single_state['input_semi_result'] = single_state['input_semi_result'][None,:]
        single_state['ptr_result'] = single_state['ptr_result'][None,:]
        single_state['inputcarry'] = single_state['inputcarry'][None,:]
        single_state['ptr_carry'] = single_state['ptr_carry'][None,:]
        predicted = self.targetModel(single_state,session)
        return predicted

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else:
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate:
            action = np.random.randint(0, self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, session, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            x = {}
            x['inputA'] = []
            x['inputB'] = []
            x['inputcarry'] = []
            x['input_semi_result'] = []
            x['ptr_carry'] = []
            x['ptr_result'] = []
            x['target_q_t'] = []
            x['action'] = []
            # X_batch = np.empty((0, self.input_size), dtype=np.float64)
            # Y_batch = np.empty((0, self.output_size), dtype=np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state, session)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else:
                    qValuesNewState = self.getQValues(newState, session)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)  # y_j

                x['inputA'].append(state['inputA'])
                x['inputB'].append(state['inputB'])
                x['inputcarry'].append(state['inputcarry'])
                x['input_semi_result'].append(state['input_semi_result'])
                x['ptr_carry'].append(state['ptr_carry'])
                x['ptr_result'].append(state['ptr_result'])
                x['target_q_t'].append(targetValue)
                x['action'].append(action)

                # X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                # Y_sample = qValues.copy()
                # Y_sample[action] = targetValue
                # Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                # if isFinal:
                #     X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                #     Y_batch = np.append(Y_batch, np.array([[reward] * self.output_size]), axis=0)

            self.model.train(session,x )
            # self.model.fit(X_batch, Y_batch, batch_size=len(miniBatch), nb_epoch=1, verbose=0)

hyperparams = {}
hyperparams['epochs'] = 1000
hyperparams['steps'] = 100000
hyperparams['updateTargetNetwork'] = 10000
hyperparams['explorationRate'] = 1
hyperparams['minibatch_size'] = 128
hyperparams['learnStart'] = 128
hyperparams['learningRate'] = 0.00025
hyperparams['discountFactor'] = 0.99

last100Scores = [0] * 100
last100ScoresIndex = 0
last100Filled = False


memorySize = 1000000
memory =  Memory(memorySize)


# deepQ = DeepQ(10, 3, memorySize, discountFactor, learningRate, learnStart)
# deepQ.initNetworks([30,30,30])
# deepQ.initNetworks([30,30])
# deepQ.initNetworks([300,300])

stepCounter = 0

config_gpu = tf.ConfigProto(allow_soft_placement=True)
config_gpu.gpu_options.allow_growth = True
# number of reruns
with tf.Session(config = config_gpu) as session:
    init = tf.global_variables_initializer()
    session.run(init)


    # monte_carlo_tree_search_training(memory, session, hyperparams)

    # Set the rounds to play
    for epochs in tqdm(range(500)):
        init_state = State(env.reset())
        init_node = Node()
        init_node.set_state(init_state)
        current_node = init_node
        statelist = []
        done = False
        for i in range(30):
            print("Play round: {}".format(i + 1))
            current_node = monte_carlo_tree_search(current_node, session)
            print("Choose node: {}".format(current_node))
            statelist.append(current_node)
            if current_node.done:
                done = True
                break

        Store_in_memory(statelist, memory, done)
        Train_Net(memory, session, hyperparams)

        TestNet(session, env)


    # for epoch in range(epochs):
    #     observation = env.reset()
    #     # print explorationRate
    #     # number of timesteps
    #     for t in range(steps):
    #         print (t,'step')
    #         print (observation)
    #         # env.render()
    #         qValues = deepQ.getQValues(observation, session)
    #
    #         action = deepQ.selectAction(qValues, explorationRate)
    #         print('action:',action)
    #
    #         newObservation, reward, done, info = env.step(action)
    #
    #         if (t >= 199):
    #             print ("reached the end! :D")
    #             done = True
    #             # reward = 200
    #
    #         if done and t < 199:
    #             print ("decrease reward")
    #             # reward -= 200
    #         deepQ.addMemory(observation, action, reward, newObservation, done)
    #
    #         if stepCounter >= learnStart:
    #             if stepCounter <= updateTargetNetwork:
    #                 print ('no target')
    #                 deepQ.learnOnMiniBatch(minibatch_size, session, False)
    #             else :
    #                 deepQ.learnOnMiniBatch(minibatch_size, session, True)
    #
    #         observation = newObservation
    #
    #         if done:
    #             last100Scores[last100ScoresIndex] = t
    #             last100ScoresIndex += 1
    #             if last100ScoresIndex >= 100:
    #                 last100Filled = True
    #                 last100ScoresIndex = 0
    #             if not last100Filled:
    #                 print ("Episode ",epoch," finished after {} timesteps".format(t+1))
    #             else :
    #                 print ("Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores)))
    #             break
    #
    #         stepCounter += 1
    #         if stepCounter % updateTargetNetwork == 0:
    #             deepQ.updateTargetNetwork()
    #             print ("updating target network")

        # explorationRate *= 0.995
        # # explorationRate -= (2.0/epochs)
        # explorationRate = max (0.05, explorationRate)