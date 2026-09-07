# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import time

#0. 평가 지표(RMSLE) 계산 함수 작성
def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test,y_predict))
    return rmse


#1. 데이터
path = "./_data/bike-sharing-demand/"          

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)            #[10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)             #[6493 rows x 8 columns]              

submission = pd.read_csv(path + "sampleSubmission.csv" , index_col=0)
# print(submission)           #[6493 rows x 1 columns]               

# print(train_csv.info()) # 결측치 없음
# print(test_csv.info())
"""
######################### 결측치 확인 ###################################
# print(train_csv.isna().sum())


######################### 결측치 처리 1. 삭제 ####################################
train_csv = train_csv.dropna()
print(train_csv)            #[1328 rows x 10 columns]
""
######################### 결측치 처리 2. 평균값 처리 ####################################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())  #(715,9)
"""

# train_csv를 x와 y로 분리
x = train_csv.drop(['casual','registered', 'count'], axis=1) #열 삭제)
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=3124)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size=0.33, random_state=1231)

#2. 모델 구성
model = Sequential()
model.add(Dense(10,activation='relu', input_dim=8))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
          verbose = 1, validation_data = (x_val,y_val))
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
print("RMSE value : ", rmse)


############### submission.csv 만들기 // 결과값을 count cloumn에 넣어준다 ##################
y_submit = model.predict(test_csv)

submission['count'] = y_submit
# print(submission)
# print(submission.shape)

# submission.to_csv(path + "submit/" + "submit_0907_1012_AF.csv")

"""
Hyperparameter tuning
file                :   submit_0907_1012_AF
remarks             :   Add activation function(Relu) at hidden layer
random_state        :   3124
train_size          :   0.7
list_percepticon    :   [10,10,10,10,10,1]
epochs              :   500
batch_size          :   8
RMSE value          :   147.9581
"""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,6))
plt.plot(np.log(hist.history['loss'][10:]), c='red', label='loss')
plt.plot(np.log(hist.history['val_loss'][10:]), c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('kaggle Loss')
plt.xlabel('epoch')
plt.ylabel('los')
plt.grid()
plt.show()