import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)

alpha = 0.01

def leaky_relu(x,alpha):
    return np.maximum(alpha*x,x)


y = leaky_relu(x,alpha)

plt.plot(x,y)
plt.grid()
plt.show()