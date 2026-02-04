import random 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(52)

def toss_coin(n):
    return np.random.choice(['Heads', 'Tails'], size=n)

"""
a=int(input("enter number of tosses: "))
results = toss_coin(a)
print(results)

heads_ratio = np.sum(results == 'Heads')/a
print(heads_ratio)"""
"""
ratios = []
for n in range(10,1000,10): #10,20,30,40 ..... 980,990,999
    resulting_coin = toss_coin(n)
    print(f"{resulting_coin} for n={n}")
    r = np.sum(resulting_coin == 'Heads')/n
    ratios.append(r)
"""

prob = 0.9
trials = np.random.rand(20) < prob
print(trials)
print("failures despite high confidence", np.sum(trials==False))