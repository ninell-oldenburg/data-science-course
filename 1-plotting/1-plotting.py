#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# dataset stats
# [0] age – a positive integer (years)
# [1] FEV1 – a continuous valued measurement (liter)
# [2] height – a continuous valued measurement (inches)
# [3] gender – binary (female: 0, male: 1)
# [4] smoking status – binary (non-smoker: 0, smoker: 1)
# [5] weight – a continuous valued measurement (kg)


# Read the data from the file smoking.txt, and divide the dataset into 
# two groups consisting of smokers and non-smokers. 

path = "smoking.txt"
data = np.loadtxt(path)
data = data.astype(float)

smoks = data[np.isin(data[:, 4], 1)]
non_smoks = data[np.isin(data[:, 4], 0)]

# Write a script which computes the average lung function, 
# measured in FEV1, among the smokers and among the non-smokers

avg_smoks = np.mean(smoks, axis=0)[1]
avg_non_smoks = np.mean(non_smoks, axis=0)[1]

print('average lung function for smokers: {}'.format(avg_smoks))
print('average lung function for non smokers: {}'.format(avg_non_smoks))

# Report your computed average FEV1 scores. Are you surprised?
# Yes, I am definitely surprised because it is expected that smokers have a smaller lung volumne in litres.

# Make a box plot of the FEV1 in the two groups. What do you see? Are you surprised?

fig, ax = plt.subplots(figsize =(10, 7))
ax.set_title('FEV1 measurement over smokers vs. non smokers')
box_data = [smoks[:, 1], non_smoks[:, 1]]
ax.set_xticklabels(['smokers', 'non smokers']);
ax.set_xlabel('group');
ax.set_ylabel('FEV1 in liters');
ax.boxplot(box_data);

# As expected form the average value in Exercise 1, the box plot supports the findings of 
# smokers having a higher lung volumne. The variance, however, seems to be higher for non smokers, 
# as well as the number of outliers.

# # Exercise 3

# ## a)

# Write a script that performs a two-sided t-test whose null hypothesis is that the two
# populations have the same mean. Use a significance level of α = 0.05, and return a
# binary response indicating acceptance or rejection of the null hypothesis. You should
# try do implement it by yourself – though not the CDF of the t-distribution, use scipy.
# If you can’t, you may use scipy’s stats.ttest ind.

from scipy import stats
from statistics import variance

def ttest_welsh(x, y, alpha):
    
    result = True
    
    df = math.floor((((variance(x)/len(x)) + (variance(y)/len(y)))**2) / ((((variance(x)/len(x))**2)/(len(x)-1)) + (((variance(y)/len(y))**2)/(len(y)-1))))
    t = (np.mean(x) - np.mean(y)) / (np.sqrt((variance(x)/len(x)) + (variance(y)/len(y))))
    
    crit_val = stats.t.ppf(q = 1-alpha, df=df)
    p_val = 2 * (1- stats.t.cdf(t, df))
    
    if p_val <= alpha:
        result = False
    
    return 't: {},\ndf: {},\np-val: {},\naccept?: {}'.format(t, df, p_val, result)

alpha = 0.05
print(ttest_welsh(smoks[:, 1], non_smoks[:, 1], alpha))

# # Exercise 4

# ## a)

# Compute the correlation between age and FEV1. Make a 2D plot of age versus FEV1
# where non smokers appear in one color and smokers appear in another.

fig, ax = plt.subplots(figsize =(10, 7))
ax.set_title('FEV1-age-correlation over smokers vs. non smokers')
plt.scatter(smoks[:, 0], smoks[:, 1], color='r', label='smokers');
plt.scatter(non_smoks[:, 0], non_smoks[:, 1], color='g', label='non smokers');
plt.legend();
ax.set_xlabel('group');
ax.set_ylabel('FEV1 in liters');

x = data[:,0]
y = data[:,1]

rkx = np.argsort(x)
rky = np.argsort(y)

print(f"Pearson Correlation coefficients matrix:\n{np.corrcoef(x,y)}")
print(f"Spearman Correlation coefficients matrix:\n{np.corrcoef(rkx,rky)}")


# # Exercise 5

# ## a)

# Create a histogram over the age of subjects in each of the two groups, smokers and non-smokers.

print(len(non_smoks[:, 0]), len(smoks[:, 0]))

fig, ax = plt.subplots(figsize =(10, 7))
ax.set_title('FEV1-age-correlation over smokers vs. non smokers')
plt.hist(non_smoks[:, 0], bins=len(set(non_smoks[:, 0])), color='g', label='non smokers');
plt.hist(smoks[:, 0], bins=len(set(smoks[:, 0])), color='r', label='smokers');
plt.legend();
plt.grid();
ax.set_xlabel('group');
ax.set_ylabel('FEV1 in liters');

# Overall, three things become even more clear than in the previous analysis.
"""
1. The number of non smokers exceeds the number of smokers drastically.
2. The distribution over the age groups is very different from smokers to non smoker is two ways:
    a) The plot supports the findings from the earlier exercise that the mean age is higher in the group of smokers.
    That is, the red plotted bars for the smokers are more towards the right side of the plot
    b) The distribution of the both groups over the parameter of age is different: where there are rather high spikes
        for the ages 9, 8 und 11 in the non smokers plot, we see a more flat curve throughout the smokers age groups.
        However, this could also be due to the general less datapoints in that group with some classes (age 9) only
        having 1 record.
3. Intersting side note: even though we have way more examples of non smokers (589 vs. 65), there still seem to be 
    more smokers at the ages of 15 and 16 than there are non-smokers.
"""


