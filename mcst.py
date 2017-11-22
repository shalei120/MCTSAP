#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import random
import numpy as np
from Memory import *
from npienv import *

from tqdm import tqdm
#
AVAILABLE_CHOICES = range(3)
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_ROUND_NUMBER = 30


env = environment(10)
f_func = f(10, 3)

class State(object):
    """
    蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
    需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
    """

    def __init__(self, obv):
        self.current_value = 0.0
        # For the first root node, the index is 0 and the game should start from 1
        self.current_round_index = 0
        self.cumulative_choices = []
        self.obv = obv
        self.policy_neural = []
        self.father_action = -1
        self.reward = None
        self.done = 0

    def get_current_value(self):
        return self.current_value

    def set_current_value(self, value):
        self.current_value = value

    def set_obv(self, obv):
        self.obv = obv

    def get_current_round_index(self):
        return self.current_round_index

    def set_current_round_index(self, turn):
        self.current_round_index = turn

    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_cumulative_choices(self, choices):
        self.cumulative_choices = choices

    def is_terminal(self):
        # The round index starts from 1 to max round number
        if self.current_round_index == MAX_ROUND_NUMBER:
            return True
        else:
            return False

    def compute_reward(self):
        return env.Cal_reward(self.obv)

    def get_next_state_with_random_choice(self,sess):
        random_choice = random.choice([choice for choice in range(3)])

        cur_obv, reward, done, _ = env.step(random_choice)

        self.policy_neural, v = f_func(cur_obv,sess)

        self.policy_neural = self.policy_neural[0]

        next_state = State(cur_obv)
        next_state.set_current_value(self.current_value + v[0,0])
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])
        next_state.father_action = random_choice
        next_state.reward = reward
        next_state.done = done

        return next_state

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        return "State: {}, value: {}, reward: {}, round: {}, choices: {}".format(
            hash(self), self.current_value, self.reward, self.current_round_index,
            self.cumulative_choices)


class Node(object):
    """
    蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
    """

    def __init__(self):
        self.parent = None
        self.children = [None] * 3

        self.visit_times = 0
        self.quality_value = 0.0

        self.state = None

    def set_state(self, state):
        # self.quality_value = state.get_current_value()   # added by Lei sha
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        if None not in self.children:
            return True
        else:
            return False

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children[sub_node.get_state().father_action] = sub_node

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.state)


def tree_policy(node,sess):
    """
    蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
    基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
    """

    # Check if the current node is the leaf node
    while node.get_state().is_terminal() == False:

        if node.is_all_expand():
            node = best_child(node, True)
        else:
            # Return the new sub node
            sub_node = expand(node,sess)
            return sub_node

    # Return the leaf node
    return node


def default_policy(node, sess):
    """
    蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。
    基本策略是随机选择Action。
    """

    # Get the state of the game
    current_state = node.get_state()

    # Run until the game over
    while current_state.is_terminal() == False:
        # Pick one random action to play and get next state
        current_state = current_state.get_next_state_with_random_choice(sess)
        if current_state.done:
            return current_state.compute_reward(), current_state.done


    final_state_reward = current_state.compute_reward()
    return final_state_reward, current_state.done


def expand(node, sess):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """

    tried_sub_node_states = [
        sub_node.get_state() for sub_node in node.get_children() if sub_node != None
    ]

    new_state = node.get_state().get_next_state_with_random_choice(sess)

    # Check until get the new state which has the different action from others
    while new_state in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice(sess)

    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)

    return sub_node


def best_child(node, is_exploration):
    """
    使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
    """

    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None

    # Travel all sub nodes to find the best one
    Nsum  = sum([sub_node.get_visit_times() for sub_node in node.get_children()])
    # print node.get_state().policy_neural
    for sub_node, p in zip(node.get_children(), node.get_state().policy_neural):

        # Ignore exploration for inference
        if is_exploration:
            C = 1 / math.sqrt(2.0)
        else:
            C = 0.0

        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        left = sub_node.get_state().current_value / sub_node.get_visit_times()
        right =   p * np.sqrt(Nsum)/(1 + sub_node.get_visit_times())
        score = left + C * np.sqrt(right)

        # print  p, left,right, score
        if score > best_score:
            best_sub_node = sub_node
            best_score = score

    return best_sub_node


def backup(node, reward, done):
    """
    蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
    """

    # Update util the root node
    while node != None:
        # Update the visit times
        node.visit_times_add_one()

        # Update the quality value
        node.quality_value_add_n(reward)
        node.done  = done

        # Change the node to the parent node
        node = node.parent

def Store_in_memory(nodelist, memo, done):
    tau = 0.8
    eps = 0.000001

    for root in nodelist:
        pi = np.asarray([node.get_visit_times() if node!=None else 0 for node in root.children], dtype = 'float32')
        pi = pi ** (1/tau)
        pi = (pi + eps) / (sum(pi) + eps * len(pi))
        memo.addMemory(root.get_state(), pi, 1 if done else 0)# 1.0 * root.get_quality_value() / root.get_visit_times())

    # for node in root.children:
    #     if node != None:
    #         Store_in_memory(node, memo)

def Train_Net(memory, sess, hyperparams):
    model = f_func
    if memory.getCurrentSize() > hyperparams['learnStart']:
        for i in range(100):
            miniBatch = memory.getMiniBatch(hyperparams['minibatch_size'])
            x = {}
            x['inputA'] = []
            x['inputB'] = []
            x['inputcarry'] = []
            x['input_semi_result'] = []
            x['ptr_carry'] = []
            x['ptr_result'] = []
            x['target_q_t'] = []
            x['pi'] = []


            for sample in miniBatch:
                obv = sample['state'].obv
                pi = sample['pi']
                reward = sample['reward']


                x['inputA'].append(obv['inputA'])
                x['inputB'].append(obv['inputB'])
                x['inputcarry'].append(obv['inputcarry'])
                x['input_semi_result'].append(obv['input_semi_result'])
                x['ptr_carry'].append(obv['ptr_carry'])
                x['ptr_result'].append(obv['ptr_result'])
                x['target_q_t'].append(reward)
                x['pi'].append(pi)

            # model.debug(sess,x)
            model.train(sess, x)


def TestNet(sess, env):
    model = f_func
    state = env.reset()
    steps = 0
    for i in range(31):

        p,v = model(state, sess)
        act = np.argmax(p)
        state, r,done, _ = env.step(act)
        print state,act,p
        steps += 1
        if done:
            break

    print steps



def monte_carlo_tree_search(node, sess):
    """
    实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。
    蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
    前两步使用tree policy找到值得探索的节点。
    第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
    最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
    进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
    """

    computation_budget = 100

    #
    # for epoch in range(500):
        # init_state = State(env.reset())
        # init_node = Node()
        # init_node.set_state(init_state)
        # node = init_nod
            # Run as much as possible under the computation budget
    for i in range(computation_budget):
        # 1. Find the best node to expand
        expand_node = tree_policy(node,sess)

        # 2. Random run to add node and get reward
        reward, done = default_policy(expand_node,sess)

        # 3. Update all passing nodes with reward
        backup(expand_node, reward, done)




        # Store_in_memory(node, memo)
        #
        # Train_Net(memo, sess, hyperparams)
        #
        # TestNet(sess, env)

        # Tree_Clear(node)

    # N. Get the best next node
    best_next_node = best_child(node, False)

    return best_next_node


# def main():
#     # Create the initialized state and initialized node
#     init_state = State()
#     init_node = Node()
#     init_node.set_state(init_state)
#     current_node = init_node
#
#     # Set the rounds to play
#     for i in range(30):
#         print("Play round: {}".format(i + 1))
#         current_node = monte_carlo_tree_search(current_node)
#         print("Choose node: {}".format(current_node))


if __name__ == "__main__":
    main()
