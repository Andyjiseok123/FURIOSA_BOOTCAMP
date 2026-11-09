# 03-00 copy
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
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

iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets=[iris,cancer,wine,digits]
data_name=['아이리스','캔서','와인','디지트']

# 2. 모델
all = all_estimators(type_filter='classifier')          # 44개
# all = all_estimators(type_filter='regressor')           # 55개
n_split = 5
kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=4123)
# print('all_models :',all)
# print('sk version : ',sk.__version__)
# print('모델의 갯수 : ',len(all))
for index, value in enumerate(datasets):
    x, y = value
    x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, train_size=0.8, random_state=4123, stratify=y)
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

    