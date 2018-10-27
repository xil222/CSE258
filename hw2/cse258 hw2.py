
# coding: utf-8

# In[1]:


import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from math import exp
from math import log
from urllib.request import urlopen
import numpy as np
import pandas as pd
import math
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[2]:


def parseData(fname):
  for l in urlopen(fname):
    yield eval(l)


# In[3]:


data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))


# In[4]:


def feature(datum):
  feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'], datum['review/overall']]
  return feat


# In[61]:


X = [feature(d) for d in data]
y = [d['beer/ABV'] >= 6.5 for d in data]


# In[6]:


def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))


# In[23]:


def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

X_train = X
y_train = y

##################################################
# Train                                          #
##################################################

def train(lam, X_train, y_train):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta, X, y_train):
  scores = [inner(theta,x) for x in X]
  predictions = [s > 0 for s in scores]
  correct = [(a==b) for (a,b) in zip(predictions,y_train)]
  acc = sum(correct) * 1.0 / len(correct)
  return acc

##################################################
# Validation pipeline                            #
##################################################

lam = 1.0

theta = train(lam, X_train, y_train)
acc = performance(theta, X_train, y_train)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc))


# In[24]:


#problem 1


# In[25]:


x1, y1 = shuffle(X, y)
X_train = x1[:int(len(x1)/3)]
X_val = x1[int(len(x1)/3):int(len(x1)*2/3)]
X_test = x1[int(len(x1)*2/3):]

y_train = y1[:int(len(y1)/3)]
y_val = y1[int(len(x1)/3):int(len(x1)*2/3)]
y_test = y1[int(len(y1)*2/3):]


# In[26]:


theta = train(lam, X_train, y_train)
acc = performance(theta, X_val, y_val)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc))


# In[27]:


acc = performance(theta, X_test, y_test)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc))


# In[ ]:


#problem 2


# In[28]:


def reportResult(theta, X, y):
    scores = [inner(theta,x) for x in X]
#     print(scores)
    predictions = [s > 0 for s in scores]
    pos, neg = 0, 0 
    tpos, tneg, fpos, fneg = 0, 0, 0, 0
    for (a,b) in zip(predictions,y):
        if a == True:
            pos += 1
            if b == True:
                tpos += 1
            else:
                fpos += 1
        else:
            neg += 1
            if b == False:
                tneg += 1
            else:
                fneg += 1
    print("Positive: " + str(pos))
    print("Negative: " + str(neg))
    print("True Positive: " + str(tpos))
    print("True Negative: " + str(tneg))
    print("False Positive: " + str(fpos))
    print("False Negative: " + str(fneg))
            


# In[29]:


reportResult(theta, X_test, y_test)


# In[30]:


#problem 3


# In[31]:


# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    if y[i]:
      loglikelihood -= log(1 + exp(-logit))
    else:
      loglikelihood -= 10 * logit + 10 * log(1 + exp(-logit))
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
    
#   if not y[i]:
#     loglikelihood *= 10
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      if y[i]:
        dl[k] += X[i][k] * (1 - sigmoid(logit))
      else:
        dl[k] += 10 * X[i][k] * (1 - sigmoid(logit))
        dl[k] -= 10 * X[i][k]
        
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
    
  return numpy.array([-x for x in dl])


# In[32]:


X_train = X
y_train = y

##################################################
# Train                                          #
##################################################

def train(lam, X_train, y_train):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta, X, y_train):
  scores = [inner(theta,x) for x in X]
  predictions = [s > 0 for s in scores]
  correct = [(a==b) for (a,b) in zip(predictions,y_train)]
  acc = sum(correct) * 1.0 / len(correct)
  return acc

##################################################
# Validation pipeline                            #
##################################################

lam = 1

theta = train(lam, X_train, y_train)
acc = performance(theta, X_train, y_train)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc))


# In[33]:


def reportResult(theta, X, y):
    scores = [inner(theta,x) for x in X]
    predictions = [s > 0 for s in scores]
    pos, neg = 0, 0 
    tpos, tneg, fpos, fneg = 0, 0, 0, 0
    for (a,b) in zip(predictions,y):
        if b == True:
            pos += 1
            if a == True:
                tpos += 1
            else:
                fneg += 1
        else:
            neg += 1
            if a == False:
                tneg += 1
            else:
                fpos += 1
    print("Positive: " + str(pos))
    print("Negative: " + str(neg))
    print("True Positive: " + str(tpos))
    print("True Negative: " + str(tneg))
    print("False Positive: " + str(fpos))
    print("False Negative: " + str(fneg))


# In[34]:


reportResult(theta, X_test, y_test)


# In[18]:


#problem 4


# In[62]:



x1, y1 = shuffle(X, y)
X_train = x1[:int(len(x1)/3)]
X_val = x1[int(len(x1)/3):int(len(x1)*2/3)]
X_test = x1[int(len(x1)*2/3):]

y_train = y1[:int(len(y1)/3)]
y_val = y1[int(len(x1)/3):int(len(x1)*2/3)]
y_test = y1[int(len(y1)*2/3):]

def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])


# In[63]:


for lam in [0, 0.01, 0.1, 1, 100]:
    print(lam)
    theta = train(lam, X_train, y_train)
    acc = performance(theta, X_train, y_train)
    print("Train :\taccuracy=" + str(acc))
    acc = performance(theta, X_val, y_val)
    print("Validation :\taccuracy=" + str(acc))
    acc = performance(theta, X_test, y_test)
    print("Test :\taccuracy=" + str(acc))
    


# In[43]:


import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


# In[44]:


X = [[x['review/overall'], x['review/taste'], x['review/aroma'], x['review/appearance'], x['review/palate']] for x in data]

pca = PCA(n_components=5)
pca.fit(X)


# In[ ]:


len(X)


# In[67]:


# Karate club
G = nx.karate_club_graph()
nx.draw(G)
# plt.show()
# plt.clf()

edges = set()
nodes = set()
for edge in urlopen("http://jmcauley.ucsd.edu/cse255/data/facebook/egonet.txt"):
  x,y = edge.split()
  x,y = int(x),int(y)
  edges.add((x,y))
  edges.add((y,x))
  nodes.add(x)
  nodes.add(y)

G = nx.Graph()
for e in edges:
  G.add_edge(e[0],e[1])
nx.draw(G)
# plt.show()
# plt.clf()

### Find all 3 and 4-cliques in the graph ###
cliques3 = set()
cliques4 = set()
for n1 in nodes:
  for n2 in nodes:
    if not ((n1,n2) in edges): continue
    for n3 in nodes:
      if not ((n1,n3) in edges): continue
      if not ((n2,n3) in edges): continue
      clique = [n1,n2,n3]
      clique.sort()
      cliques3.add(tuple(clique))
      for n4 in nodes:
        if not ((n1,n4) in edges): continue
        if not ((n2,n4) in edges): continue
        if not ((n3,n4) in edges): continue
        clique = [n1,n2,n3,n4]
        clique.sort()
        cliques4.add(tuple(clique))


# In[68]:


#problem 5


# In[69]:


nx.number_connected_components(G)


# In[70]:


for i in nx.connected_components(G):
    print(len(i))


# In[71]:


#problem 6
maxLen = 1
newgraph = G
for c in nx.connected_components(G):
    if G.subgraph(c).number_of_nodes() > maxLen:
        maxLen = G.subgraph(c).number_of_nodes()
        res = c
        newgraph = G.subgraph(c)


# In[72]:


nx.node_connected_component(G, list(res)[0])


# In[73]:


a = list(sorted(res))
length = len(a) // 2
set1 = set(a[:length])
set2 = set(a[length:])


# In[74]:


print(nx.normalized_cut_size(newgraph,set1,set2)/2)


# In[75]:


# print(nx.cut_size(G,set1,set2))


# In[82]:


# problem 7
# set1, set2

#1. try move set1 to set2
#2. try move set2 to set1
#3  move the smallest if cost is lower

def normalize_cost(set1, set2, G):
    curr_cost = nx.normalized_cut_size(G,set1,set2)/2
    while True:
        listA = list(set1)
        listB = list(set2)
        
        eleA, costA = findMin(listA, set1, set2, G)
        eleB, costB = findMin(listB, set2, set1, G)
        
        if min(costA, costB) >= curr_cost:
            break
        else:
            if costA < costB:
                listA.remove(eleA)
                listB.append(eleA)
                set1 = set(listA)
                set2 = set(listB)
                curr_cost = costA
            else:
                listB.remove(eleB)
                listA.append(eleB)
                set1 = set(listA)
                set2 = set(listB)
                curr_cost = costB
    
        print(curr_cost)
    print(sorted(set1))
    print(sorted(set2))
    return set1, set2, curr_cost


# In[83]:


#find the minVal in listA, and cost
def findMin(tempList, setA, setB, G):
    minCost = nx.normalized_cut_size(G,setA,setB)/2
    ele = 0
    for i in tempList:
        temp1 = list(setA)
        temp2 = list(setB)
        temp1.remove(i)
        temp2.append(i)
        cur_cost = nx.normalized_cut_size(G,set(temp1),set(temp2))/2
        if cur_cost < minCost:
            ele = i
            minCost = cur_cost
    return ele, minCost
        
        


# In[84]:


normalize_cost(set1,set2,G)


# In[85]:


#problem 8

#1. try move set1 to set2
#2. try move set2 to set1
#3  move the one if modularity is greater

def normalize_modularity(G, set1, set2, community):
    curr_modularity = nx.algorithms.community.modularity(G, community)
    while True:
        listA = list(set1)
        listB = list(set2)
        
        eleA, modA = findMax(listA, community, set1, set2, G)
        eleB, modB = findMax(listB, community, set2, set1, G)
        
        if max(modA, modB) <= curr_modularity:
            break
        else:
            if modA > modB:
                listA.remove(eleA)
                listB.append(eleA)
                community = list([set(listA),set(listB)])
                set1 = community[0]
                set2 = community[1]
                curr_modularity = modA
            else:
                listB.remove(eleB)
                listA.append(eleB)
                community = list([set(listA),set(listB)])
                set1 = community[0]
                set2 = community[1]
                curr_modularity = modB
                
    print(sorted(set1))
    print(sorted(set2))
    return community, curr_modularity


# In[86]:


#find the minVal in listA, and cost
def findMax(templist, community, setA, setB, G):
    maxCost = nx.algorithms.community.modularity(G, community)
    ele = 0
    for i in templist:
        temp1 = list(setA)
        temp2 = list(setB)
        temp1.remove(i)
        temp2.append(i)
        community = list([set(temp1),set(temp2)])
        cur_cost = nx.algorithms.community.modularity(G, community)
        if cur_cost > maxCost:
            ele = i
            maxCost = cur_cost
    return ele, maxCost


# In[87]:


currSet = list([set1, set2])
normalize_modularity(newgraph, set1, set2, currSet)

