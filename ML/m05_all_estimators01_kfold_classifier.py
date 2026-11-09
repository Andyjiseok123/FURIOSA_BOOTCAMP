# 03-00 copy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x,y = load_iris(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, train_size=0.8, random_state=4123, stratify=y)
n_split=10
kfold = KFold(n_splits=n_split,shuffle=True, random_state=333) 
# print(x,y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
all = all_estimators(type_filter='classifier')          # 44개
# all = all_estimators(type_filter='regressor')           # 55개

# print('all_models :',all)
# print('sk version : ',sk.__version__)
# print('모델의 갯수 : ',len(all))

# scores = cross_val_score(model,x_train,y_train,cv=kfold)

# print("acc: ",scores,"\n평균 acc: ",round(np.mean(scores),2))

for name,model in all:
    try:
        model=model()

        scores = cross_val_score(model,x_train,y_train,cv=kfold)

        # model.fit(x_train,y_train)

        # acc = model.score(x_test,y_test)
        print(name,"의 정답률 : ",round(np.mean(scores),2))
    except:
        print(name, '모델 오류')