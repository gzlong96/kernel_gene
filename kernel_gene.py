import random
from functools import cmp_to_key
import time
import config
from agent.agent_gym import AGENT_GYM
from agent.simple_agent import Agent
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Input, Flatten, Conv2DTranspose, Reshape, Concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import keras.callbacks
import copy
import multiprocessing as mp

random.seed(1234567)
length = len(config.Map.hole_pos)
city_dis = config.Map.city_dis
np_city_dis = np.array(city_dis)
hole_city = config.Map.hole_city

config.Type_num = len(city_dis)+1
config.Source_num = len(config.Map.source_pos)
config.Hole_num = len(config.Map.hole_pos)

# dis = open("distance50.txt",'r')
# distance = eval(dis.readline())

# sta_reward = open("sta_reward.txt",'w')
# true_reward = open("true_reward.txt",'w')


trans = np.ones((config.Map.Width, config.Map.Height, 4), dtype=np.int8)
for i in range(0, config.Map.Width):
    for j in range(0, config.Map.Height):
        if i % 2 == 0:
            trans[i][j][1] = 0
        else:
            trans[i][j][3] = 0
        if j % 2 == 0:
            trans[i][j][2] = 0
        else:
            trans[i][j][0] = 0

agent = Agent()

# def get_reward_outside(hole_city):
#     agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
#                           hole_city, city_dis, trans)
#     agent_gym.reset()
#     # print mazemap
#     return agent.test(agent_gym)


# model = keras.models.load_model('models/hot_model.h5')
# def get_reward_from_net(hole_city):
#     pred_hot, pred_r = model.predict([np.reshape(to_categorical(hole_city, 5), (-1, 20, 5))], batch_size=32)
#     return pred_r[0][0]


sim_average = {}
sim_count = 0
class OneAssign:
    def __init__(self, kernel=None, rate=None):
        if kernel is None:
            self.kernel = np.random.rand(25)
        else:
            self.kernel = kernel
        if rate is None:
            self.rate = np.random.random()
        else:
            self.rate = rate
        self.reward = 0

        self.scaled_kernel = np.reshape(self.kernel*self.rate/(1-self.rate), [5,5])

    def get_reward(self):
        self.reward = self.get_reward_from_agent()
        # self.reward = self.get_reward_from_distance()
        # self.reward = get_reward_from_net(self.hole_city)
        return self.reward

    def get_reward_from_agent(self):
        agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                              hole_city, city_dis, self.scaled_kernel, trans)
        agent_gym.reset()
        # print mazemap
        r = agent.test(agent_gym)
        global sim_count
        sim_count+=1
        if str(self.scaled_kernel) not in sim_average.keys():
            if r>5500:
                r1 = agent.test(agent_gym)
                r2 = agent.test(agent_gym)
                sim_average[str(self.scaled_kernel)] = [r+r1+r2, 3]
                sim_count+=2
            else:
                sim_average[str(self.scaled_kernel)] = [r, 1]
        else:
            sim_average[str(self.scaled_kernel)][0] += r
            sim_average[str(self.scaled_kernel)][1] += 1
        return sim_average[str(self.scaled_kernel)][0]/sim_average[str(self.scaled_kernel)][1]

    # def get_reward_from_distance(self):
    #     mins = np.zeros((config.Source_num, len(city_dis)))
    #     for i in range(config.Source_num):
    #         for j in range(config.Hole_num):
    #             new_dis = distance[i][j]
    #             if mins[i][self.hole_city[j]] == 0 or new_dis<mins[i][self.hole_city[j]]:
    #                 mins[i][self.hole_city[j]] = new_dis
    #         mins[i] *= np_city_dis
    #     return 9999 - np.sum(mins)


def assign_cmp(a, b):
    if a.reward < b.reward:
        return 1
    elif a.reward == b.reward:
        return 0
    return -1


class Evolution:
    def __init__(self, seq):
        self.step = 0
        self.mutate_rate = 0.25
        self.nums = 100
        self.end = False
        self.group = [OneAssign() for _ in range(self.nums)]
        self.end_step = 100

        self.time_stamp = time.time()
        for assign in self.group:
            assign.get_reward()

        self.seq_num = str(seq)
        self.sim_reward = open("result/sim_reward_"+self.seq_num+".txt", 'w')
        self.best_assign = open("result/best_assign_"+self.seq_num+".txt", 'w')

        self.current_best = None

        # self.pool = mp.Pool(10)

    def sort_group(self):
        self.group.sort(key=cmp_to_key(assign_cmp))

    def reproduce(self):
        self.step += 1
        self.duplicate = set()
        for i in range(self.nums):
            self.group[i].get_reward()
            self.duplicate.add(str(self.group[i].scaled_kernel))
        for i in range(2*self.nums):
            father = random.randint(0, self.nums-1)
            mother = random.randint(0, self.nums-1)
            self.group.append(self.cross(father, mother))

        self.sort_group()
        if self.step == self.end_step:
            self.end = True
        # select the bests
        self.group = self.group[0:self.nums]
        print("step: " + str(self.step) + "best: " + str(self.group[0].reward))
        # print("best assign:" + str(self.group[0].hole_city))
        assigns = open("result/top_assigns_"+self.seq_num+".txt",'w')
        for i in range(5):
            if isinstance(self.group[i].hole_city, type([])):
                assigns.write(str(self.group[0].hole_city)+'\n')
            else:
                assigns.write(str(self.group[0].hole_city.tolist()) + '\n')
        print(str(self.group[99].reward))

        if self.step%1 == 0 or self.step == 1:
            if self.current_best is None or self.current_best != self.group[0]:
                self.current_best = self.group[0]
                self.sim_reward.write(str(5100) + '\n')
            print("--------------------------------------------------------")
            # true_reward.write(str(self.group[0].get_reward_from_agent())+'\n')
            # sta_reward.write(str(self.group[0].reward) + '\n')
            print(self.group[0].hole_city)
            print(sim_average[str(self.group[0].hole_city)])
            print(self.group[0].reward)
            self.sim_reward.write(str([self.group[0].get_reward(),sim_count,(time.time()-self.time_stamp)/60]) + '\n')
            self.best_assign.write(str(self.group[0].reward) + '\n')
            self.sim_reward.close()
            self.best_assign.close()
            self.sim_reward = open("result/sim_reward_"+self.seq_num+".txt", 'a')
            self.best_assign = open("result/best_assign_"+self.seq_num+".txt", 'a')


    def cross(self, father, mother):
        cross_pos = random.randint(1, 23)
        son = OneAssign(np.concatenate([self.group[father].kernel[:cross_pos], self.group[mother].kernel[cross_pos:]]),
                        (self.group[father].rate + self.group[mother].rate)/2)
        # if str(son.hole_city) not in self.duplicate:
        #     self.duplicate.add(str(son.hole_city))
        #     break

        mutate = random.random()
        if mutate < self.mutate_rate:
            for i in range(5):
            # for i in range(max([5, 10 * int(0.2 > (self.group[0].reward - self.group[-1].reward))])):
                mutate_pos = random.randint(0, 24)
                # son.hole_city[mutate_pos] = np.random.multinomial(1, city_dis, size=1).tolist()[0].index(1)
                son.kernel[mutate_pos] = (son.kernel[mutate_pos]+np.random.random())/2
            son = OneAssign(son.kernel, son.rate * 0.9 + np.random.random() * 0.1)
        son.get_reward()
        return son

    def run(self):
        time11 = time.time()
        while not self.end:
            time22 = time.time()
            self.reproduce()
            print("One round:", (time.time() - time22)/60.0)
            print("Total:", (time.time() - time11)/3600.0)


def one_evo(i):
    evo = Evolution(i)
    evo.run()



if __name__ == "__main__":
    # time1 = time.time()
    # pl = mp.Pool(10)
    # for i in range(10):
    #     pl.apply_async(one_evo,[i])
    # pl.close()
    # pl.join()

    one_evo(0)

    # time2 = time.time()
    # print("total time: " + str((time2-time1)/3600.0))
