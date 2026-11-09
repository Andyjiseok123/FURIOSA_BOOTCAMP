import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 데이터
x,y = load_iris(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, train_size=0.8, random_state=4123, stratify=y)
# print(x,y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split=10
kfold = KFold(n_splits=n_split,shuffle=True, random_state=4123)         #Class

# 2. 모델
model = DecisionTreeClassifier()

# 3,4. 훈련
scores = cross_val_score(model,x_train,y_train,cv=kfold)

print("acc: ",scores,"\n평균 acc: ",round(np.mean(scores),2))

"""
acc:  [1.         0.93333333 0.93333333 0.93333333 0.93333333 0.93333333
 1.         0.86666667 1.         0.93333333] 
평균 acc:  0.95
"""\

"""
acc:  [0.91666667 1.         1.         0.91666667 0.91666667 0.91666667
 0.91666667 0.83333333 0.83333333 0.83333333] 
평균 acc:  0.91
"""

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)
acc_score = accuracy_score(y_test,y_predict)
print(acc_score)