
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
openPrice   = np.array(data.open)
closePrice  = np.array(data.close)
high        = np.array(data.high)
low         = np.array(data.low)
volume      = np.array(data.volume)
time        = np.array(data.time)

# In[4]:


def switch(price,buckets):# decide which state the price belongs to
    temp = [0]*len(buckets)    
    for i in range(len(buckets)-1):
        if price >= buckets[i] and price < buckets[i+1]:
            return i

def transition(state, action):
    newState = state.clone()
    if action < 0:
        newState.buyCoin(-action)
    elif action > 0:
        newState.sellCoin(action)
        
    newState.advanceTimeStep()
    return newState

INITIAL_INVESTMENT = 1000000
TRANSACTION_FEE = 0.0025    
        
class BitcoinState:
    
    #marketPrice: array of last n market prices
    #marketVolume: array of last n market volume
    #marketHigh: array of last n market highs
    #marketLow: array of last n market lows
    def __init__(self, timeIndex, timeRange=60, dollar=50000.0, coin=0.0):
        self.timeIndex = timeIndex
        self.timeRange = timeRange
        self.dollar = dollar
        self.coin = coin
        
    
    def buyCoin(self, dollar):
        self.dollar -= dollar
        self.coin += (1-TRANSACTION_FEE) * dollar / closePrice[self.timeIndex]
        
    def sellCoin(self, dollar):
        self.dollar += (1-TRANSACTION_FEE) * dollar
        self.coin -= dollar / closePrice[self.timeIndex]
        
    def netWorth(self):
        return self.dollar + self.coin * closePrice[self.timeIndex]
    
    def advanceTimeStep(self):
        self.timeIndex += 1
    
    def clone(self):
        return BitcoinState(self.timeIndex, self.timeRange, self.dollar, self.coin)
            
    def isTerminal(self):
        return self.netWorth() <= 0
    
    def isValid(self):
        return self.timeIndex >= self.timeRange \
            and self.timeIndex < len(time) \
            and self.dollar >= 0 \
            and self.coin >= 0
    
    def getMarketHighs(self):
        return high[self.timeIndex - self.timeRange : self.timeIndex]
    
    def getMarketLows(self):
        return low[self.timeIndex - self.timeRange : self.timeIndex]
    
    def getMarketOpenPrices(self):
        return openPrice[self.timeIndex - self.timeRange : self.timeIndex]
    
    def getMarketClosePrices(self):
        return closePrice[self.timeIndex - self.timeRange : self.timeIndex]
    
    def getMarketVolumes(self):
        return volume[self.timeIndex - self.timeRange : self.timeIndex]
    
    def __str__(self):
        return '{' + \
            'timeIndex:' + str(self.timeIndex) + ',' + \
            'timeRange:' + str(self.timeRange) + ',' + \
            'dollar:' + str(self.dollar) + ',' + \
            'coin:' + str(self.coin) + \
        '}'
        
    
    
def bitcoinFeatureExtractor(state, action):
      
    newState = transition(state, action)
    
    #Extract asset info
    features = [
        #("month" , month),
        #("day"   , day),
        #("hour"  , hour),
        #("minute", minute),
        #("dollar", newState.dollar),
        #("coin"  , newState.coin),
        #("coinWorth", newState.coin * closePrice[newState.timeIndex])
        ("percentageWorth", newState.netWorth() / INITIAL_INVESTMENT),
        ("percentageGrowth", (newState.netWorth()-INITIAL_INVESTMENT) / INITIAL_INVESTMENT)
    ]
    
    #Extract time info
    timeStr = datetime.fromtimestamp(time[newState.timeIndex]).strftime('%Y-%m-%d %H:%M:%S')
    month   = int(timeStr[5:7])
    day     = int(timeStr[8:10])
    hour    = int(timeStr[11:13])
    minute  = int(timeStr[14:16])
    for k, v in getMonthOneHotVector(month).items():
        features.append((k, v))
    
    #Extract market info
    for i, price in enumerate(newState.getMarketOpenPrices()):
        features.append(("openPrice-" + str(i), price))
    
    for i, price in enumerate(newState.getMarketClosePrices()):
        features.append(("closePrice-" + str(i), price))
        
    for i, volume in enumerate(newState.getMarketVolumes()):
        features.append(("volume-" + str(i), volume))
        
    for i, high in enumerate(newState.getMarketHighs()):
        features.append(("high-" + str(i), high))
        
    for i, low in enumerate(newState.getMarketLows()):
        features.append(("low-" + str(i), low))    
        
    return features    

def getMonthOneHotVector(month):
    return {
        "January"   : int(month == 1),
        "February"  : int(month == 2),
        "March"     : int(month == 3),
        "April"     : int(month == 4),
        "May"       : int(month == 5),
        "June"      : int(month == 6),
        "July"      : int(month == 7),
        "August"    : int(month == 8),
        "September" : int(month == 9),
        "October"   : int(month == 10),
        "November"  : int(month == 11),
        "December"  : int(month == 12)
    }

def getPossibleActions(state):
    possibleActions = []
    for action in action_space:
        successor = transition(state, action)
        if successor.isValid():
            possibleActions.append(action)
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
    
    updateInterval = max(numTrials / 100000, 10)
    
    for trial in range(numTrials):
        
        if trial % updateInterval == 0:
            print("Completion:", "%0.3f" % (trial/numTrials*100),"%", end="\r", flush=True)
       
        totalReward = 0
#         totalReward_percentage = 1
        #print(len(price),len(time))
        startTimeIndex = random.randint(time_range, len(time) - time_range - 1)
        while time[startTimeIndex + time_range] - time[startTimeIndex] != time_range * 60 \
            and time[startTimeIndex] - time[startTimeIndex - time_range] != time_range * 60:
           startTimeIndex = random.randint(time_range, len(time) - time_range - 1)
            #current = switch(price[startTimeIndex],buckets)
        
        state = BitcoinState(
            timeIndex=startTimeIndex - 1,
            timeRange = time_range,
            dollar=INITIAL_INVESTMENT,
            coin=0.0
        )
        #print("Worth:",state.netWorth(),"Price:",price[startTimeIndex], "Time:",time[startTimeIndex])
        
        for _ in range(time_range):
            #print("DEBUG: State=",state)
            #get action based on exploration policy
            # ACTION FOR Q LEARNING
            
            action = qla.getAction(state)     #get a random action
            successor = transition(           #apply action
                state=state,
                action=action,
            )
            
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
    LEARNING_TRIALS = 100000
    TESTING_TRIALS = 100
    
    print("Start Learning (" + str(LEARNING_TRIALS) + " trials)")

    #Learn with epsilon = 0.5
    #qla = QLearningAlgorithm(actions = sawyer.getActions, discount = 1, explorationProb = 0.8)
    qla = QLearningApproxAlgorithm(actions = getPossibleActions, discount = 1, featureExtractor=bitcoinFeatureExtractor, explorationProb = 0.7)
    #qla.load()

    totalRewards = myQLearning(qla, numTrials=LEARNING_TRIALS, time_range=time_range, test=False)
    ##print("Total Rewards:", totalRewards)

    print("")
    print("Finish Learning")

    ##Act optimally by setting epsilon = 0
    qla.explorationProb = 0
    totalRewards = myQLearning(qla, numTrials=TESTING_TRIALS, time_range=time_range,test=True)
    print("Total Rewards:", totalRewards)
    print("Average Rewards:", sum(totalRewards)/TESTING_TRIALS)
    print("Finish Testing")
    print("Q-Learning Completed")
    #print("Number of States explored:", len(qla.Q))
    
    print("Top 20 Weight")
    for w in sorted(qla.weights.items(), key=lambda x : x[1], reverse=True)[:20]:
        print(w[0], ":", w[1])



main()

