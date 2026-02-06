import numpy as np
import matplotlib.pyplot as plt

predictions = np.random.beta(5,2,1000 )
print(predictions)
plt.hist(predictions, bins=30, density=True)
plt.title ("Model confidence distribution")
#plt.show()

confident = predictions > 0.8
print("Highly confident predictions : ", confident.mean())
print("total confident predictions : ", confident.sum())