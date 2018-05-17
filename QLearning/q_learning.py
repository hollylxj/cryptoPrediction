
# coding: utf-8

# In[3]:


import os
from math import pi
import time
import random
from operator import add
import pickle
import pandas
import numpy as np
from datetime import datetime
#import ql
import math
from qla import QLearningApproxAlgorithm

Inf = math.inf

with open('../BTC-USD-60.pkl', 'rb') as f:
    data = pickle.load(f)

#:chunk = filter_df(df_chunk, event_type='Fill')
# sort the data based on ascending-order of time
data = data.sort_values(by=['time'])
price = np.array(data.close)
time = np.array(data.time)


# In[4]:


def switch(price,buckets):# decide which state the price belongs to
    temp = [0]*len(buckets)    
    for i in range(len(buckets)-1):
        if price >= buckets[i] and price < buckets[i+1]:
            return i

def transition(state, action, price, time):
    newState = state.clone(state.marketPrice, time)
    if action < 0:
        newState.sellCoin(-action)
    elif action > 0:
        newState.buyCoin(action)
    newState.marketPrice = price
    return newState
    
        
class BitcoinState:
    def __init__(self, marketPrice, time, dollarInvestment=1000, coin=0):
        self.dollar = dollarInvestment
        self.coin = coin
        self.marketPrice = marketPrice
        time = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
        self.month = int(time[5:7])
        self.day = int(time[8:10])
        self.hour = int(time[11:13])
        self.minute = int(time[14:16])
        
    def buyCoin(self, quantity):
        self.dollar -= self.marketPrice * quantity
        self.coin += quantity
        
    def sellCoin(self, quantity):
        self.dollar += self.marketPrice * quantity
        self.coin -= quantity
        
    def netWorth(self):
        return self.dollar + self.coin * self.marketPrice
    
    def clone(self, marketPrice, time):
        return BitcoinState(marketPrice, time, self.dollar, self.coin)
            
    def isTerminal(self):
        return self.netWorth() <= 0
        
    
    
def bitcoinFeatureExtractor(state, action):
    return [
        ("price" , state.marketPrice),
        ("month" , state.month),
        ("day"   , state.day),
        ("hour"  , state.hour),
        ("minute", state.minute),
        #("dollar", state.dollar),
        ("coin"  , state.coin),
        ("coinWorth", state.coin * state.marketPrice)
    ]

# Recursively get bucket separating values
def bucket_separate(nBuckets,bucket_list,y_distribution):
    mid = len(y_distribution)//2 # median index of input y_distribution list
    if nBuckets % 2 == 1:
        print("number of buckets must be 2^n")

    elif nBuckets == 2:
        bucket_list.append(y_distribution[mid])

    else:
        bucket_list.append(y_distribution[mid])
        bucket_list = bucket_separate(nBuckets/2,bucket_list,y_distribution[:mid])#left
        bucket_list = bucket_separate(nBuckets/2,bucket_list,y_distribution[mid:])#right
    bucket_list = sorted(bucket_list)

    return bucket_list


# In[5]:


action_space = [-10, 0, 10]#[-1000, -100, -10, 0, 10, 100, 1000]
#price_distribution = sorted(price)
#nBuckets = 1024
#buckets = bucket_separate(nBuckets,[-Inf,0,Inf],price_distribution)


# In[16]:


def myQLearning(qla, numTrials=1000, time_range=60, verbose=False, test = False):
    #logging initializations
        
    totalDiscount = 1
    totalRewards = []

    for trial in range(numTrials):
       
        totalReward = 0
#         totalReward_percentage = 1
        #print(len(price),len(time))
        startTimeIndex = random.randint(0, len(price) - 61)
        while  time[startTimeIndex + time_range] - time[startTimeIndex] != time_range * 60:
           startTimeIndex = random.randint(0, len(price) - 61)
            #current = switch(price[startTimeIndex],buckets)
        
        state = BitcoinState(marketPrice=price[startTimeIndex], time=time[startTimeIndex], dollarInvestment=1000, coin=0)
        #print("Worth:",state.netWorth(),"Price:",price[startTimeIndex], "Time:",time[startTimeIndex])
        
        for i in range(startTimeIndex+1, startTimeIndex+time_range):
            #print("DEBUG: State=",state)
            #get action based on exploration policy
            # ACTION FOR Q LEARNING
            
            action = qla.getAction(state)     #get a random action
            successor = transition(state, action, price=price[i], time=time[i])          #apply action
            reward = successor.netWorth() - state.netWorth()        #calculate reward
#             reward_percentage = np.log(successor.netWorth() / state.netWorth())
            #print("Dollar:",successor.dollar,"Coin:",successor.coin,"Worth:",successor.netWorth(),"Price:",successor.marketPrice, "Time:",time[i], "Action:",action,"reward:",reward)
            
            #print("DEBUG: Action=",action)
            #successor = switch(state+action,buckets)
            
            #current += 1
            #print("DEBUG: Succ=",successor)
            
            #timer += 1
            #terminalState = (timer//60 == 0)

            #if not terminalState:
            #    reward = price[i] - price[i-1]

            totalReward += reward
#             totalReward_percentage *= reward_percentage
            if successor.isTerminal():
                qla.incorporateFeedback(state, action, reward, None)
                break;
            qla.incorporateFeedback(state, action, reward, successor)

            state = successor

        if verbose:
            print("Trial %d totalReward = %s" % (trial, totalReward))
            
        totalRewards.append(totalReward)

        
    return totalRewards


# In[ ]:


def main():
    time_range = 60
    LEARNING_TRIALS = 2000000
    TESTING_TRIALS = 1500
    


    #Learn with epsilon = 0.5
    #qla = QLearningAlgorithm(actions = sawyer.getActions, discount = 1, explorationProb = 0.8)
    qla = QLearningApproxAlgorithm(actions = lambda state : action_space, discount = 1, featureExtractor=bitcoinFeatureExtractor, explorationProb = 0.5)
    #qla.load()

    totalRewards = myQLearning(qla, numTrials=LEARNING_TRIALS, time_range=time_range, test=False)
    ##print("Total Rewards:", totalRewards)

    print("Finish Learning")

    ##Act optimally by setting epsilon = 0
    qla.explorationProb = 0
    totalRewards = myQLearning(qla, numTrials=TESTING_TRIALS, time_range=time_range,test=True)
    print("Total Rewards:", totalRewards)
    print("Average Rewards:", sum(totalRewards)/TESTING_TRIALS)
    print("Finish Testing")
    print("Q-Learning Completed")
    #print("Number of States explored:", len(qla.Q))



main()

