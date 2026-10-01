import numpy as np

sin1 = 3.5/5
cos1 = 3/5

cof = np.sqrt(sin1**2+cos1**2)

sin2 = sin1/cof
cos2 = cos1/cof

rad1 = np.arctan2(sin1 ,cos1)
rad2 = np.arctan2(sin2 ,cos2)
print(np.rad2deg(rad1),np.rad2deg(rad2))