import numpy as np
import tensorflow as tf
import random

class environment(object):

    def __init__(self, input_size):
        self.input_size = input_size
        self.reset()

    def randgen(self, len):
        res = []
        for i in range(len):
            res.append(random.randint(0,9))

        return res
    def arr2int(self, arr):
        res=0
        for d in arr:
            res = res * 10 + d
        return res

    def int2arr(self, num, length):
        arr = []
        while num > 0:
            arr.insert(0, num%10)
            num = num/10

        while len(arr) < length:
            arr.insert(0,0)

        return arr
    def arr_add(self, a,b, length):
        res = []
        c = 0
        for i in range(len(a))[::-1]:
            res.insert(0, (a[i] + b[i] + c)%10)
            c= (a[i] + b[i]+c)//10
        return res

    def reset(self):
        self.inputA = np.asarray([0, 2, 4, 8, 9, 4, 2, 4, 5, 5, 9]) #np.asarray([0] + self.randgen(self.input_size))
        self.inputB = np.asarray([0, 5, 4, 1, 6, 5, 9, 6, 3, 4, 7]) # np.asarray([0] + self.randgen(self.input_size))
        self.inputcarry = np.asarray([0]*(self.input_size+1))
        self.input_semi_result = np.asarray([0] * (self.input_size+1))
        self.ptr_carry = np.asarray([0]*(self.input_size) + [1] ) # 0000100000
        self.ptr_result = np.asarray([0]*(self.input_size) + [1])
        self.sum = self.arr_add(self.inputA, self.inputB, length = self.input_size + 1)
        print('sum',self.sum)

        self.obv = {}
        self.obv['input_size'] = self.input_size
        self.obv['inputA'] = self.inputA
        self.obv['inputB'] = self.inputB
        self.obv['inputcarry'] = self.inputcarry
        self.obv['input_semi_result'] = self.input_semi_result
        self.obv['ptr_carry'] = self.ptr_carry
        self.obv['ptr_result'] = self.ptr_result
        return  self.obv

    def current_observation(self):
        obv = {}
        obv['input_size'] = self.input_size
        obv['inputA'] = self.inputA
        obv['inputB'] = self.inputB
        obv['inputcarry'] = self.inputcarry
        obv['input_semi_result'] = self.input_semi_result
        obv['ptr_carry'] = self.ptr_carry
        obv['ptr_result'] = self.ptr_result
        return obv

    def step(self, action):
        if action == 0:
            self.Add()
        elif action == 1:
            self.Carry()
        elif action == 2:
            self.MovePtr()

        # print (len(self.sum), len(self.input_semi_result))
        valid = True
        for ans,semi in zip(self.sum, self.input_semi_result):
            valid = valid and (ans == semi)

        reward = self.Cal_reward(self.obv)
        done = True if valid else False

        return self.current_observation(), reward, done, 0

    def Cal_reward(self, obv):
        diff = 1.0*abs(self.sum - obv['input_semi_result'])
        diff += 0.1
        diff = 1.0/diff
        return sum(diff)

    def Add(self):
        pos = np.argmax(self.ptr_carry)
        res =  self.inputA[pos]+ self.inputB[pos] + self.input_semi_result[pos]
        self.input_semi_result[pos] = res % 10
        self.inputcarry[pos] = res /10

    def Carry(self):
        pos = np.argmax(self.ptr_carry)
        if pos >= len(self.inputcarry)-1:
            return

        self.input_semi_result[pos] = self.inputcarry[pos+1]
        self.inputcarry[pos + 1] = 0

        if self.input_semi_result[pos] > 10:
            self.inputcarry[pos] += self.input_semi_result[pos] / 10
            self.input_semi_result[pos] = self.input_semi_result[pos] % 10

    def MovePtr(self):
        pos = np.argmax(self.ptr_carry)
        pos = max(pos-1, 1)
        self.ptr_carry = np.asarray([0]*(self.input_size+1))
        self.ptr_carry[pos] = 1
