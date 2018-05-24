
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
price  = np.array(data.close)
high   = np.array(data.high)
low    = np.array(data.low)
volume = np.array(data.volume)
time   = np.array(data.time)

# In[4]:


def switch(price,buckets):# decide which state the price belongs to
    temp = [0]*len(buckets)    
    for i in range(len(buckets)-1):
        if price >= buckets[i] and price < buckets[i+1]:
            return i

def transition(state, action, price, volume, high, low, time):
    newState = state.clone(state.marketPrice, state.marketVolume, state.marketHigh, state.marketLow, time)
    if action < 0:
        newState.buyCoin(-action)
    elif action > 0:
        newState.sellCoin(action)
        
    #throw least recent entry and append most recent entry
    newState.marketPrice  = np.append(newState.marketPrice[1:], price)
    newState.marketVolume = np.append(newState.marketVolume[1:], volume)
    newState.marketHigh   = np.append(newState.marketHigh[1:], high)
    newState.marketLow    = np.append(newState.marketLow[1:], low)
    return newState
    
        
class BitcoinState:
    
    #marketPrice: array of last n market prices
    #marketVolume: array of last n market volume
    #marketHigh: array of last n market highs
    #marketLow: array of last n market lows
    def __init__(self, marketPrice, marketVolume, marketHigh, marketLow, time, dollarInvestment=50000.0, coin=0.0):
        self.dollar = dollarInvestment
        self.coin = coin
        self.marketPrice  = marketPrice
        self.marketVolume = marketVolume
        self.marketHigh   = marketHigh
        self.marketLow    = marketLow
        time = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
        self.month = int(time[5:7])
        self.day = int(time[8:10])
        self.hour = int(time[11:13])
        self.minute = int(time[14:16])
    
    def buyCoin(self, dollar):
        self.dollar -= dollar
        self.coin += dollar / self.marketPrice[-1]
        
    def sellCoin(self, dollar):
        self.dollar += dollar
        self.coin -= dollar / self.marketPrice[-1]
        
    def netWorth(self):
        return self.dollar + self.coin * self.marketPrice[-1]
    
    def clone(self, marketPrice, marketVolume, marketHigh, marketLow, time):
        return BitcoinState(marketPrice, marketVolume, marketHigh, marketLow, time, self.dollar, self.coin)
            
    def isTerminal(self):
        return self.netWorth() <= 0
        
    
    
def bitcoinFeatureExtractor(state, action):
    features = [
        ("month" , state.month),
        ("day"   , state.day),
        ("hour"  , state.hour),
        ("minute", state.minute),
        ("dollar", state.dollar),
        ("coin"  , state.coin),
        ("coinWorth", state.coin * state.marketPrice[-1]),
        ("action", action)
    ]
    for i, price in enumerate(state.marketPrice):
        features.append(("price-" + str(i), price))
        
    for i, volume in enumerate(state.marketVolume):
        features.append(("volume-" + str(i), volume))
        
    for i, high in enumerate(state.marketHigh):
        features.append(("high-" + str(i), high))
        
    for i, low in enumerate(state.marketLow):
        features.append(("low-" + str(i), low))    
        
    return features    
    
def getPossibleActions(state):
    possibleActions = []
    for changeInDollar in action_space:
        if -changeInDollar <= state.dollar and state.coin >= changeInDollar / state.marketPrice[-1]:
            possibleActions.append(changeInDollar)
    return possibleActions
    
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

#change in dollars
action_space = [-1000.0, 0.0, 1000.0]#[-1000, -100, -10, 0, 10, 100, 1000]

#price_distribution = sorted(price)
#nBuckets = 1024
#buckets = bucket_separate(nBuckets,[-Inf,0,Inf],price_distribution)


# In[16]:


def myQLearning(qla, numTrials=1000, time_range=60, verbose=False, test = False):
    #logging initializations
        
    totalDiscount = 1
    totalRewards = []
    
    updateInterval = numTrials/100000
    
    for trial in range(numTrials):
        
        if trial % updateInterval == 0:
            print("Completion:", "%0.3f" % (trial/numTrials*100),"%", end="\r", flush=True)
       
        totalReward = 0
#         totalReward_percentage = 1
        #print(len(price),len(time))
        startTimeIndex = random.randint(time_range, len(price) - time_range - 1)
        while time[startTimeIndex + time_range] - time[startTimeIndex] != time_range * 60 \
            and time[startTimeIndex] - time[startTimeIndex - time_range] != time_range * 60:
           startTimeIndex = random.randint(time_range, len(price) - time_range - 1)
            #current = switch(price[startTimeIndex],buckets)
        
        state = BitcoinState(
            marketPrice=price[startTimeIndex-time_range:startTimeIndex],
            marketVolume=volume[startTimeIndex-time_range:startTimeIndex],
            marketHigh=high[startTimeIndex-time_range:startTimeIndex],
            marketLow=low[startTimeIndex-time_range:startTimeIndex],
            time=time[startTimeIndex - 1],
            dollarInvestment=50000.0,
            coin=0.0
        )
        #print("Worth:",state.netWorth(),"Price:",price[startTimeIndex], "Time:",time[startTimeIndex])
        
        for i in range(startTimeIndex, startTimeIndex+time_range):
            #print("DEBUG: State=",state)
            #get action based on exploration policy
            # ACTION FOR Q LEARNING
            
            action = qla.getAction(state)     #get a random action
            successor = transition(           #apply action
                state=state,
                action=action,
                price=price[i],
                volume=volume[i],
                high=high[i],
                low=low[i],
                time=time[i]
            )
            
            if(successor.marketPrice.size != time_range):
                print(i)
                print(successor.marketPrice.size)
                print(successor.marketHigh.size)
                print(successor.marketLow.size)
                print(successor.marketVolume.size)
                exit()
            
            reward = successor.netWorth() - state.netWorth()        #calculate reward

            totalReward += reward

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
    LEARNING_TRIALS = 1000000
    TESTING_TRIALS = 1000
    
    print("Start Learning")

    #Learn with epsilon = 0.5
    #qla = QLearningAlgorithm(actions = sawyer.getActions, discount = 1, explorationProb = 0.8)
    qla = QLearningApproxAlgorithm(actions = getPossibleActions, discount = 1, featureExtractor=bitcoinFeatureExtractor, explorationProb = 0.7)
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
    print("Weights", qla.weights)



main()

