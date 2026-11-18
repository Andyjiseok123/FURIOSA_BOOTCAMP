import pandas as pd
import numpy as np

data = pd.DataFrame([[2,np.nan,6,8,10],
                    [2,4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan,4,np.nan,8,np.nan],
                    ]).T
data.columns=['x1','x2','x3','x4']
# print(data)

#0. 결측치
# print(data.isnull())
# print(data.isnull().sum)

#1. 결측치 삭제
print(data.dropna(axis=1)) #axis 옵션 0:행삭제, 1:열삭제

#2-1. 특정값 - 평균
means = data.mean()
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
med = data.median()
data3 = data.fillna(med)
print(data3)

#2-3. 특정값 - 0
data4 = data.fillna(0)
print(data4)

#2-4. 특정값 - ffill (통상 마지막 값), (시계열)
data5 = data.ffill()
print(data5)

#2-5. 특정값 - bfill (통상 첫번째 값), (시계열)
data6 = data.bfill()
print(data6)

print("=========================================")
#################### 특정 칼럼만 #############################
means = data['x1'].mead()
