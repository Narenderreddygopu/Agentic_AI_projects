import random 
import numpy as np

np.random.seed(36)

users = 1000
premium = np.random.rand(users) < 0.3  # 30% premium users
clicked = np.where(premium, np.random.rand(users) < 0.7, np.random.rand(users) < 0.2)

p_click = clicked.mean()
p_click_given_premium = clicked[premium].mean()

print("p=(click) : ",round(p_click,2))
print("p=(click|premium) : ",round(p_click_given_premium,2))


# no. of users = 1000
# premium users = 300
# non-premium users = 700
# total no. of clicks = 0.34*1000 = 340
# total no. premium users clicked from the total no. of clicks = 0.67 * 340 = 227 

def likelihood(obs, prob):
    return np.prod([prob if o else (1-prob) for o in obs])
observations = [True, True, False, True]
print("likelihood (prob = 0.8) : ", likelihood(observations, 0.8))
print("likelihood (prob = 0.2) : ", likelihood(observations, 0.2))

"""

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
"""