from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

#1.데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=221,
    train_size=0.75, 

)
from sklearn.preprocessing import MinMaxScaler,StandardScaler ,MaxAbsScaler #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler
##############################################################################
# scaler = MinMaxScaler()
##############################################################################



##############################################################################
# scaler = StandardScaler()
##############################################################################



##############################################################################
# scaler = MaxAbsScaler()
##############################################################################


##############################################################################
scaler = RobustScaler()
##############################################################################

scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.fit_transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,))



#3.컴파일,훈련
from keras.optimizers import Adam
model.compile(loss='mse', optimizer= Adam(learning_rate=0.01))
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True
)
hist = model.fit(x_train,y_train, 
                 epochs=3000, 
                 batch_size=10 ,
                 validation_split=0.2,
                 callbacks =[es]
                 )


#4.평가,예측
print("=========================================")

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)                                                                                                                                  

r2 = r2_score(y_test, y_predict) 
print('r2결과값: ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 
# results = model.predict(x)
# print('결과값: ' ,results)


'''
하이퍼 파라미터 튜닝
#1.데이터 부분
random_state
train_size
#2.
레이어의 깊이
노드의갯수
#3
epoch
batch_size
'''


"""
1차시도
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2637.873291015625
r2결과값:  0.5483979249211794
mse :  2637.8732195134085
RMSE :  51.36022994023107
"""

"""
2차시도 --- MINMAX-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2574.98388671875
r2결과값:  0.5591645248939214
mse :  2574.983947518029
RMSE :  50.744299655409854
"""



"""
3차시도 --- standard-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2451.989013671875
r2결과값:  0.5802211882185655
mse :  2451.9889230450026
RMSE :  49.5175617639338
"""

"""
3차시도 --- standard-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2661.037353515625
r2결과값:  0.5444322418537058
mse :  2661.0373494810156
RMSE :  51.58524352449076
"""

"""
r2결과값:  0.5356532721306715
mse :  2712.3165849080656
RMSE :  52.0799057690014       
"""