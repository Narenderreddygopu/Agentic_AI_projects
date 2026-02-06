import random 
import numpy as np

true = np.random.rand(1000) < 0.7 # 0.67, 01
print(true)
#pred_conf = np.random.uniform(0.5,0.9, 1000)  #for equal values
pred_conf = np.random.uniform(0.6,1.0, 1000)   # 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,...
accuracy = np.mean(true)
avg_conf = pred_conf.mean()

print("accuracy : ",round(accuracy,3))
print("average confidence : ", round(avg_conf,2))   