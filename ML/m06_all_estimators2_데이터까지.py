# 03-00 copy
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터

california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)


datasets=[california,diabetes]
data_name=['캘리포니아','다이아비티']

# 2. 모델
all = all_estimators(type_filter='regressor')          # 44개
# all = all_estimators(type_filter='regressor')           # 55개
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=4123)

for index, value in enumerate(datasets):
    x, y = value
    x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, train_size=0.8, random_state=4123)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("=================", data_name[index],"===================")
    model_name=[]
    model_acc=[]

    for name,model in all:
        try:
            model = model()

            scores = cross_val_score(model,x_train,y_train,cv=kfold)

            y_pred = cross_val_predict(model,x_test,y_test, cv=kfold)

            acc = np.round(accuracy_score(y_test,y_pred),4)
            model_name.append(name)
            model_acc.append(acc)
            
        except:
            # print(name, '모델 오류')
            continue
    model_name=np.array(model_name)
    model_acc=np.array(model_acc)
    max_index = np.argmax(model_acc)

    print('최고 성능 모델 : ', model_name[max_index], 'ACC : ',model_acc[max_index])

    