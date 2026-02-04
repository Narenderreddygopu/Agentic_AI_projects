import numpy as np


def like(obs, prob) : 
    em_l = []
    for ele in obs : 
        if ele == True :
            em_l.append(prob)
        else : 
            em_l.append(1-prob)
    multipy = 1
    for i in em_l : 
        multipy = multipy * i
    return multipy

observations = [True, True, False, True]
print("like (prob = 0.8) : ", like(observations, 0.8))
print("like (prob = 0.2) : ", like(observations, 0.2))