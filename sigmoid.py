import matplotlib.pylab as plt
import numpy as np

w = 5
b = 4
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x*w + b))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()