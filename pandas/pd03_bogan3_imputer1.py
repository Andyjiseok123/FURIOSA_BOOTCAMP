import pandas as pd
import numpy as np

data = pd.DataFrame([[2,np.nan,6,8,10],
                    [2,4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan,4,np.nan,8,np.nan],
                    ]).T
data.columns=['x1','x2','x3','x4']

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer

imputer = SimpleImputer()   # 디폴트로 mean
data2 = imputer.fit_transform(data)
# print(data2)

imputer2 = SimpleImputer(strategy='mean')
data3 = imputer2.fit_transform(data)
# print(data3)

imputer3 = SimpleImputer(strategy='median')
data4 = imputer3.fit_transform(data)
print(data4)