#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import relevant packages
import numpy as np

# read in the data
dataTrain = np.loadtxt('OccupancyTrain.csv', delimiter=',')
dataTest = np.loadtxt('OccupancyTest.csv', delimiter=',')

# split input variable & labels
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]
XTest = dataTest[:,:-1]
YTest = dataTest[:,-1]


# # Exercise 1 (Nearest neighbor classification). 

# In[2]:


# simple euclidean distance function
def euclidean_dis(v1, v2):
    s = 0.0
    if len(v1) == len(v2):
        for i in range(len(v1)):
            s += (v1[i] - v2[i])**2
    return np.sqrt(s)


# In[3]:


# function to determine the label of ONE input 
# for a given train set, train label set, and k
def k_nn(train_data, train_labels, test_entry, k):
    
    dist = np.empty(len(train_data))
    for i, entry in enumerate(train_data):
        dist[i] = euclidean_dis(test_entry, entry)
        
    idx = np.argpartition(dist, k)
    neighbors_label = train_labels[idx[:k]].astype(int)
    label = neighbors_label[0]
    if not len(neighbors_label) == 1:  
        label = np.bincount(neighbors_label).argmax()
    
    return label


# In[4]:


# fit function that returns a list with the predicted class
# for a given test data set and k (call k_nn function)
def fit(x_train, y_train, x_test, k):
    prediction = []
    for i, test in enumerate(x_test):
        prediction.append(k_nn(x_train, y_train, test, k))
    return prediction


# In[5]:


# print accuracy of the above fit() function
from sklearn.metrics import accuracy_score
accTest = accuracy_score(YTest, fit(XTrain, YTrain, XTest, 1))
accTrain = accuracy_score(YTrain, fit(XTrain, YTrain, XTrain, 1))
print(f'Test accuracy (k={1}): {accTest}\nTrain accuracy (k={1}): {accTrain}')


# # Exercise 2 (Cross-validation). 

# In[6]:


from sklearn.model_selection import KFold

def get_kbest(ks, train_data, split=5):
    # create indices for CV
    cv = KFold(n_splits=split)
    # loop over CV folds
    acc_score = np.zeros((len(ks)))
    for i, k in enumerate(ks):
        for train, test in cv.split(train_data):
            XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train], XTrain[test], YTrain[train], YTrain[test]
            acc_score[i] += 1 - (accuracy_score(YTestCV, fit(XTrainCV, YTrainCV, XTestCV, k)))
        acc_score[i] = acc_score[i] / split
    
    idx_max = np.argmin(acc_score)
    kbest = ks[idx_max]
    
    return kbest

k_list = [1, 3, 5, 7, 9, 11]
kb = get_kbest(k_list, XTrain)
print('Cross-validation of XTrain:\nkbest: {}'.format(kb))


# # Exercise 3 (Evaluation of classification performance). 

# In[7]:


# import timeit
from timeit import default_timer as timer

start = timer()
accTest = accuracy_score(YTest, fit(XTrain, YTrain, XTest, kb))
end = timer()
accTrain = accuracy_score(YTrain, fit(XTrain, YTrain, XTrain, kb))
print(f'Test accuracy (k={kb}): {accTest}\nTrain accuracy (k={kb}): {accTrain}\nComputation Time: {round(end-start, 2)} sec')


# # Exercise 4 (Data normalization). 

# In[8]:


from sklearn import preprocessing

# version 1
# CORRECT VERSION 
scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)
kb = get_kbest(k_list, XTrainN)
print('VERSION 1')
print('kbest: {}'.format(kb))
start = timer()
accTest = accuracy_score(YTest, fit(XTrainN, YTrain, XTestN, kb))
end = timer()
accTrain = accuracy_score(YTrain, fit(XTrainN, YTrain, XTrainN, kb))
print(f'Test accuracy: {accTest}\nTrain accuracy: {accTrain}\nComputation Time: {round(end-start, 2)} sec')


# In[9]:


# version 2
scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
scaler = preprocessing.StandardScaler().fit(XTest)
XTestN = scaler.transform(XTest)
kb = get_kbest(k_list, XTrainN)
print('VERSION 2')
print('kbest: {}'.format(kb))
start = timer()
accTest = accuracy_score(YTest, fit(XTrainN, YTrain, XTestN, kb))
end = timer()
accTrain = accuracy_score(YTrain, fit(XTrainN, YTrain, XTrainN, kb))
print(f'Test accuracy: {accTest}\nTrain accuracy: {accTrain}\nComputation Time: {round(end-start, 2)} sec')


# In[ ]:


# version 3
XTotal = np.concatenate((XTrain, XTest))
scaler = preprocessing.StandardScaler().fit(XTotal)
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)
kb = get_kbest(k_list, XTrainN)
print('VERSION 3')
print('kbest: {}'.format(kb))
start = timer()
accTest = accuracy_score(YTest, fit(XTrainN, YTrain, XTestN, kb))
end = timer()
accTrain = accuracy_score(YTrain, fit(XTrainN, YTrain, XTrainN, kb))
print(f'Test accuracy: {accTest}\nTrain accuracy: {accTrain}\nComputation Time: {round(end-start, 2)} sec')

