import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 1. 데이터
path = './_data/kaggle_santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0 )

n_split=10
kfold = KFold(n_splits=n_split,shuffle=True, random_state=4132)         #Class

# 2. 모델
model = RandomForestRegressor()

# 3,4. 훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("acc: ",scores,"\n평균 acc: ",round(np.mean(scores),2))

# 평균 acc:  0.84                   # xgboost
# 평균 acc:  0.81                   # randomforest
