#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv

# dataset stats
# [0] age – a positive integer (years)
# [1] FEV1 – a continuous valued measurement (liter)
# [2] height – a continuous valued measurement (inches)
# [3] gender – binary (female: 0, male: 1)
# [4] smoking status – binary (non-smoker: 0, smoker: 1)
# [5] weight – a continuous valued measurement (kg)


# Exercise 1
# a)

# Read the data from the file smoking.txt, and divide the dataset into 
# two groups consisting of smokers and non-smokers. 

with open("smoking.txt", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter="\t")
    data = list(reader)
data = np.array(data)
data = data.astype(float)

smoks = data[np.isin(data[:, 4], 1)]
non_smoks = data[np.isin(data[:, 4], 0)]

# Write a script which computes the average lung function,
# measured in FEV1, among the smokers and among the non-smokers

avg_smoks = np.mean(smoks, axis=0)[1]
avg_non_smoks = np.mean(non_smoks, axis=0)[1]

print('average lung function for smokers: {}'.format(avg_smoks))
print('average lung function for non smokers: {}'.format(avg_non_smoks))

# Exercise 2
# Make a box plot of the FEV1 in the two groups. What do you see? Are you surprised?

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('FEV1 measurement over smokers vs. non smokers')
data = [smoks[:, 1], non_smoks[:, 1]]
ax.set_xticklabels(['smokers', 'non smokers']);
ax.set_xlabel('group');
ax.set_ylabel('FEV1 in liters');
plt.grid();
ax.boxplot(data);
plt.show();

# Exercise 3
# a)

# Write a script that performs a two-sided t-test whose null hypothesis is that the two
# populations have the same mean. Use a significance level of α = 0.05, and return a
# binary response indicating acceptance or rejection of the null hypothesis. You should
# try do implement it by yourself – though not the CDF of the t-distribution, use scipy.
# If you can’t, you may use scipy’s stats.ttest ind.


from scipy import stats


def t_test_same_mean(data1, data2, alpha):
    result = True
    n1 = len(data1)
    n2 = len(data2)
    df = n1 + n2 - 2

    std_dev1 = np.std(data1)
    std_dev2 = np.std(data2)

    gr_std_diff = ((n1 - 1) * (std_dev1 ** 2)) + ((n2 - 1) * (std_dev2 ** 2))
    gr_std_diff = np.sqrt((gr_std_diff / df))

    t = (np.mean(data1) - np.mean(data2)) / (gr_std_diff * np.sqrt((1 / n1) + (1 / n2)))

    if t >= stats.t.ppf(q=1 - .05 / 2, df=df):
        result = False

    return 't: {},\ndf: {},\np-val: {},\naccept?: {}'.format(t, df, alpha, result)


p = 0.05
print("=======\nT-test:\n=======\n{}".format(t_test_same_mean(smoks[:, 1], non_smoks[:, 1], p)))

# Exercise 4
# a)

# Compute the correlation between age and FEV1. Make a 2D plot of age versus FEV1
# where non smokers appear in one color and smokers appear in another.

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('FEV1-age-correlation over smokers vs. non smokers')
plt.plot(sorted(smoks[:, 0]), sorted(smoks[:, 1]), color='r', label='smokers');
plt.plot(sorted(non_smoks[:, 0]), sorted(non_smoks[:, 1]), color='g', label='non smokers');
plt.legend();
plt.grid();
ax.set_xlabel('group');
ax.set_ylabel('FEV1 in liters');
plt.show();

# Exercise 5
# a)

# Create a histogram over the age of subjects in each of the two groups, smokers and non-smokers.

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('FEV1-age-correlation over smokers vs. non smokers')
plt.hist(non_smoks[:, 0], bins=len(set(non_smoks[:, 0])), color='g', label='non smokers');
plt.hist(smoks[:, 0], bins=len(set(smoks[:, 0])), color='r', label='smokers');
plt.legend();
plt.grid();
ax.set_xlabel('group');
ax.set_ylabel('FEV1 in liters');
plt.show();