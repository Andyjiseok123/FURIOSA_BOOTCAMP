# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
path = "./_data/santander-customer-transaction-prediction/"          

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)            #[20000 rows x 201 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)             #[20000 rows x 200 columns]

submission = pd.read_csv(path + "sample_Submission.csv" , index_col=0)
# print(submission)           #[20000 rows x 1 columns]   

x=train_csv.drop(['target'],axis=1)
y=train_csv['target']
# print(x)
# print(y)

# 0과 1의 개수가 몇개인지 찾아보기 - numpy 기능
# print(np.unique(y))                         #[0,1]
# print(np.unique(y,return_counts=True))      #(array([0, 1]), array([212, 357]))

# 0과 1의 개수가 몇개인지 찾아보기 - pandas 기능
# print(pd.DataFrame(y).value_counts())       #0 179902 1 20098

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.85, 
                                                    random_state=338671,
                                                    stratify=y)                 #stratify = y : 0과 1 의 비율에 맞추어 train,test split

#2. 모델 구성
model = Sequential()
model.add(Dense(314, activation='relu', input_dim=200))
model.add(Dense(800, activation='relu'))
model.add(Dense(1824, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc']
              )                                           # metrics : 주요 보조지표

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=200,
    restore_best_weights=True
    )


start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs = 5000,
                 batch_size = 1000,
                 verbose = 1,
                 validation_split = 65/85,
                 callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("소요시간 : ", round(end_time-start_time), "초")
print("loss : " ,round(loss[0],4))
print("acc : " ,round(loss[1],4))

y_pred = model.predict(x_test)
acc_score = accuracy_score(y_test,np.round(y_pred))
print("acc_score : ", acc_score)

############### submission.csv 만들기 // 결과값을 count cloumn에 넣어준다 ##################
y_submit = model.predict(test_csv)

submission['target'] = y_submit
# print(submission)
# print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0908_1641.csv")

"""
loss :  0.2636
acc :  0.9072
"""


# print("============================ hist =============================")
# print(hist)
# print("============================ hist.history =============================")
# print(hist.history)
# print("============================ loss =============================")
# print(hist.history['loss'])
# print("============================ val_loss =============================")
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('cancer Loss')
plt.xlabel('epoch')
plt.ylabel('los')
plt.grid()
plt.show()