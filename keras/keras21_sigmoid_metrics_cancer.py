import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_breast_cancer             #유방암 관련 데이터

#1 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(y)
# 0과 1의 개수가 몇개인지 찾아보기 - numpy 기능
print(np.unique(y))                         #[0,1]
print(np.unique(y,return_counts=True))      #(array([0, 1]), array([212, 357]))

# 0과 1의 개수가 몇개인지 찾아보기 - pandas 기능
print(pd.DataFrame(y).value_counts())       #1 357 0 212
print(pd.Series(y).value_counts())          #1 357 0 212

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.85, 
                                                    random_state=1423,
                                                    stratify=y)                 #stratify = y : 0과 1 의 비율에 맞추어 train,test split

#2. 모델 구성
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=30))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics = ['accuracy']
            metrics=['acc']
              )                                           # metrics : 주요 보조지표

es = EarlyStopping(
    monitor='acc',
    mode='max',
    patience=100,
    restore_best_weights=True
    )


start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs = 5000,
                 batch_size = 15,
                 verbose = 1,
                 validation_split = 70/85,
                 callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : " ,round(loss[0],4))
print("acc : " ,round(loss[1],4))


print("소요시간 : ", round(end_time-start_time), "초")

y_pred = model.predict(x_test)
# print(round(y_pred))
# exit()

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,np.round(y_pred))
print("acc_score : ", acc_score)

# print("============================ hist =============================")
# print(hist)
# print("============================ hist.history =============================")
# print(hist.history)
# print("============================ loss =============================")
# print(hist.history['loss'])
# print("============================ val_loss =============================")
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('cancer Loss')
# plt.xlabel('epoch')
# plt.ylabel('los')
# plt.grid()
# plt.show()