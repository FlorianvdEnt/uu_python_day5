#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


print ('a. Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable')
mu = 0.6
x = np.linspace(-10, 10, 21)

dist = stats.poisson(mu)

fig, ax = plt.subplots(3, 1)
ax[0].plot(x, dist.pmf(x))
ax[1].plot(x, dist.cdf(x))
ax[2].hist(dist.rvs(1000), bins=x)
plt.savefig('poisson.png')

print ('b. Create a continious random variable with normal distribution and plot its probability mass function (PMF), cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable')
mu = 0.6
x = np.linspace(-10, 10, 21)

dist = stats.norm(mu)

fig, ax = plt.subplots(3, 1)
ax[0].plot(x, dist.pdf(x))
ax[1].plot(x, dist.cdf(x))
ax[2].hist(dist.rvs(1000))
plt.savefig('normal.png')

print ('c. Test if two sets of (independent) random data comes from the same distribution')
dist1 = stats.norm(0.5)
dist2 = stats.norm(0.5)

sample1 = dist1.rvs(200) + 1
sample1b = dist1.rvs(200) + 1
sample2 = dist2.rvs(200)

fig, ax = plt.subplots(1, 1)
ax.hist(sample1)
ax.hist(sample2)
plt.savefig('samples.png')

print('Sample 1 mu 0.5 mean 1')
print('Sample 2 mu 0.5 mean 0')
print(stats.ttest_ind(sample1, sample2))

print('\nSample 1 mu 0.5 mean 1')
print('Sample 2 mu 0.5 mean 1')
print(stats.ttest_ind(sample1, sample1b))

