import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#0. 평가 지표(RMSE) 계산 함수 작성
def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test,y_predict))
    return rmse


#1. 데이터
path = "./_data/ddarung/"                                       #상대경로
#path = "C:/Users/Admin/Desktop/jiseokLee/Project/ddarung"      #절대경로 ,  '/' '\' 둘다 상관 없음

train_csv = pd.read_csv(path + "train.csv", index_col=0)  #index_col = 0 : 첫번째 coloumn을 index 처리하여 data로 취급하지 않음
print(train_csv)            #[1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)             #[715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv" , index_col=0)
print(submission)           #[715 rows x 1 columns]

print(train_csv.info()) # pandas로 불러온 data의 information ex) 결측치 개수 등을 보여줌
print(test_csv.info())

# exit()
######################### 결측치 처리 1. 삭제 ####################################
train_csv = train_csv.dropna()
print(train_csv)            #[1328 rows x 10 columns]

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) #열 삭제)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=915124)

######################### 결측치 처리 2. 평균값 처리 ####################################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())  #(715,9)


# exit()
#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(27))
model.add(Dense(13))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1500, batch_size=50 )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
print("RMSE value : ", rmse)


############### submission.csv 만들기 // 결과값을 count cloumn에 넣어준다 ##################
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0904_1345.csv")
