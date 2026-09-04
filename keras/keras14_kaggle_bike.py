# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error

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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3124)

#2. 모델 구성
model = Sequential()
model.add(Dense(32,activation='relu', input_dim=8))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=50 )

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

submission.to_csv(path + "submit/" + "submit_0904_1639_AF.csv")

"""
Hyperparameter tuning
random_state        :   915124
train_size          :   0.85
list_percepticon    :   [100,100,32,17,9,1]
epochs              :   500
batch_size          :   50
RMSE value          :   155.349
"""
"""
Hyperparameter tuning
random_state        :   3124
train_size          :   0.7
list_percepticon    :   [32,16,8,1]
epochs              :   500
batch_size          :   50
RMSE value          :   153.334
"""
"""
Hyperparameter tuning
remarks             :   Add activation function(Relu) at last dense layer
random_state        :   3124
train_size          :   0.7
list_percepticon    :   [32,16,8,1]
epochs              :   500
batch_size          :   50
RMSE value          :   260.44
"""
"""
Hyperparameter tuning
remarks             :   Add activation function(Relu) at hidden layer & sigmoid at last layer
random_state        :   3124
train_size          :   0.7
list_percepticon    :   [32,16,8,1]
epochs              :   500
batch_size          :   50
RMSE value          :   260.44
"""