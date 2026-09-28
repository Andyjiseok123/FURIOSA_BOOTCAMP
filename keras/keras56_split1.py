import numpy as np
from my_lib import split_x
a=np.array(range(1,11))
size = 3

x,y = split_x(a,size)

print(x.shape)
print(y.shape)