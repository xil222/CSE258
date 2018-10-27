
# coding: utf-8

# In[1]:


import urllib
import scipy.optimize
import random
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
    for l in urllib.request.urlopen(fname):
        yield eval(l)


# In[3]:


data = list(parseData("http://jmcauley.ucsd.edu/cse258/data/beer/beer_50000.json"))


# In[4]:


#problem 1


# In[5]:


taste = [d['review/taste'] for d in data]


# In[6]:


plt.hist(taste)


# In[7]:


#problem 2


# In[8]:


style = [d['beer/style'] == 'Hefeweizen' for d in data]
ABV = [d['beer/ABV'] for d in data]


# In[47]:


# taste is 50000 * 1
# theta is 3 * 1
# input matrix is 50000 * 3
x1 = []
for i in style:
    if i == True:
        x1.append(1)
    else:
        x1.append(0)

x0 = np.ones((len(x1),1))

x1 = np.asarray(x1).reshape((len(x1),1))    
ABV = np.asarray(ABV).reshape((50000,1))
y = np.asarray(taste).reshape((50000,1))

x = np.hstack((x0, x1, ABV))    
    


# In[48]:


x = np.matrix(x)
y = np.matrix(y)


# In[49]:


theta = np.linalg.inv(x.T *  x) * x.T * y


# In[50]:


print(theta)


# In[51]:


#problem 3


# In[52]:


x_train = x[:int(len(x)/2)]
x_test = x[int(len(x)/2):]

y_train = y[:int(len(y)/2)]
y_test = y[int(len(y)/2):]


# In[53]:


theta_train = np.linalg.inv(x_train.T *  x_train) * x_train.T * y_train


# In[54]:


# x_train = x[:10000]
# x_test = x[10000:]

# y_train = y[:10000]
# y_test = y[10000:]
# theta_train = np.linalg.inv(x_train.T *  x_train) * x_train.T * y_train


# In[55]:



# MSE for train and test
def calculateError(theta_train, x_train, x_test, y_train, y_test):
    MSE_train = 0
    for i in range(len(x_train)):
        MSE_train += math.pow(x_train[i] * theta_train - y_train[i], 2) * 1.0 / len(x_train)
    MSE_test = 0
    for i in range(len(x_test)):
        MSE_test += math.pow(x_test[i] * theta_train - y_test[i], 2) * 1.0 / len(x_test)
    
    return MSE_train, MSE_test


# In[56]:


calculateError(theta_train, x_train, x_test, y_train, y_test)


# In[57]:


#problem 4
# c = list(zip(x,y))
# random.shuffle(c)


# In[58]:


# x,y = zip(*c)


# In[59]:


train_E = []
test_E = []

for i in range(10):
    x1, y1 = shuffle(x, y)
    x_train = x1[:int(len(x1)/2)]
    x_test = x1[int(len(x1)/2):]
    y_train = y1[:int(len(y1)/2)]
    y_test = y1[int(len(y1)/2):]
    theta_train = np.linalg.inv(x_train.T *  x_train) * x_train.T * y_train
    print(theta_train)
    train_error, test_error = calculateError(theta_train, x_train, x_test, y_train, y_test)
    train_E.append(train_error)
    test_E.append(test_error)

print(np.mean(train_E), np.mean(test_E))


# In[60]:


#problem 5


# In[61]:


# x0 --> 1, x1 --> , x2 --> 
x1 = []
x2 = []
for i in style:
    if i == True:
        x1.append(1)
        x2.append(0)
    else:
        x1.append(0)
        x2.append(1)
x1 = np.asarray(x1).reshape((len(x1),1))    
x2 = np.asarray(x2).reshape((len(x2),1))

x = np.hstack((x0, np.multiply(x1,ABV), np.multiply(x2,ABV)))    
x = np.matrix(x)


# In[62]:


train_E = []
test_E = []

for i in range(10):
    x1, y1 = shuffle(x, y)
    x_train = x1[:int(len(x1)/2)]
    x_test = x1[int(len(x1)/2):]
    y_train = y1[:int(len(y1)/2)]
    y_test = y1[int(len(y1)/2):]
    theta_train = np.linalg.inv(x_train.T *  x_train) * x_train.T * y_train
    print(theta_train)
    train_error, test_error = calculateError(theta_train, x_train, x_test, y_train, y_test)
    train_E.append(train_error)
    test_E.append(test_error)

print(np.mean(train_E), np.mean(test_E))


# In[63]:


#problem 6


# In[64]:


#problem 7


# In[65]:


taste = [d['review/taste'] for d in data]
appearance = [d['review/appearance'] for d in data]
aroma = [d['review/aroma'] for d in data]
palate = [d['review/palate'] for d in data]
overall = [d['review/overall'] for d in data]
style = [d['beer/style'] == 'Hefeweizen' for d in data]

y = []
for i in style:
    if i == True:
        y.append(1)
    else:
        y.append(-1)

taste = np.asarray(taste).reshape((len(taste),1))    
appearance = np.asarray(appearance).reshape((len(appearance),1))
aroma = np.asarray(aroma).reshape((len(aroma),1))    
palate = np.asarray(palate).reshape((len(palate),1))
overall = np.asarray(overall).reshape((len(overall),1))    
y = np.asarray(y).reshape((len(y),1))
x = np.hstack((taste, appearance, aroma,palate,overall))    


# In[66]:


x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)


# In[47]:


x = np.hstack((taste, appearance, aroma, palate, overall))
x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)


# In[41]:


x = np.hstack((taste, appearance, aroma, palate, overall))
x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)


# In[ ]:


#problem 8


# In[36]:


x = np.hstack((x0, taste, appearance, aroma, palate, overall, ABV))
x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)


# In[38]:


x = np.hstack((taste, appearance, aroma, palate, overall, ABV))
x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)


# In[39]:


x = np.hstack((taste, appearance, aroma, palate))
x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)


# In[40]:


x = np.hstack((taste, appearance, aroma, palate))
x1, y1 = shuffle(x, y)
x_train = x[:int(len(x1)/2)]
x_test = x1[int(len(x1)/2):]
y_train = y1[:int(len(y1)/2)]
y_test = y1[int(len(y1)/2):]

clf = svm.SVC(C=1000, kernel='linear')
clf.fit(x_train, y_train)

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
count = 0
for i in range(len(test_predictions)):
    if train_predictions[i] == y_train[i]:
        count += 1

acc = count/len(train_predictions)
print(acc)

count = 0
for i in range(len(train_predictions)):
    if test_predictions[i] == y_test[i]:
        count += 1

acc = count/len(test_predictions)
print(acc)

