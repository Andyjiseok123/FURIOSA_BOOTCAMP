import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import time

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=4123, train_size=0.8,stratify=y
)

n_split=5
kfold = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=4123)

parameters = [
    {'n_estimators' : [100, 500],'max_depth':[6,10,12],'learning_rate':[0.1,0.01,0.001]},
    {'max_depth':[6,8,10,12],'learngin_rate':[0.1,0.01,0.001]},
    {'min_child_weight':[2,3,5,10],'learning_rate':[0.1,0.01,0.001]}
] # 총 42

#2. 모델
xgb = XGBClassifier()
model = RandomizedSearchCV(xgb, parameters, cv=kfold, verbose=1, n_jobs=-1, random_state=4123, refit=False)     # n_jobs = cpu 사용 갯수

#3. 훈현
start = time.perf_counter()
model.fit(x_train,y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ",model.best_params_)

#4. 평가 예측
print('model score : ',model.score(x_test,y_test))
print('best score : ',model.best_score_)

print('걸린시간 : ', round(end-start,2),'초')
