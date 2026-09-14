###이진 븐류를 다중분류로 변형해서 테스트해보기###
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import os
import datetime
path = './_save/keras31/santander/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:4f}.keras'
filepath = "".join([path,"k31_",date,filename])

#1.데이터 
path = './/_data//santander-customer-transaction-prediction//'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0 )
# print(test_csv)
submission_csv = pd.read_csv(path +'sample_submission.csv',index_col=0)

x = train_csv.drop(['target'], axis=1)  #타겟 열 한개뺴기
# # print(x) #[10886 rows x 8 columns]

y = train_csv['target']
# print(x.shape,y.shape)  #(200000, 200) (200000,)
# print(y)
# print(np.unique(y,return_counts=True)) 
 #(array([0, 1]), array([179902,  20098]))
##################원 핫1, to_categorycal####################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(x.shape,y.shape)  #(200000, 200) (200000,)
# print(y)
# print(np.unique(y,return_counts=True)) 
#  #(array([0, 1]), array([179902,  20098]))
# # exit()



x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=23,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler 
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



# #2.모델구성
model = Sequential()
model.add(Dense(100, input_dim=200, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(300,activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(400, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation= 'softmax'))

#3.컴파일,훈련
model.compile(loss ='categorical_crossentropy',
                optimizer= 'adam',
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossetropy' 고정
es = EarlyStopping(
            monitor='val_loss',
            mode= 'auto',
            patience=100,
            restore_best_weights=True,
            )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath = filepath,
                      verbose=1
                      )
strat_time = time.time()  #현재 시간을 반환 ,시작시간
model.fit(x_train,y_train,
           epochs=100 , 
           batch_size=40000,
           validation_split=0.2,
           callbacks =[es,mcp], 
           )
end_time = time.time()  #현재 시간을 반환 ,시작시간

#4.평가,예측
loss = model.evaluate(x_test,y_test)#이벨류에이트 할떄는 자체적으로 라운드처리해서 acc가 나옴
print("loss:", loss)
print('걸린시간 :',round(end_time - strat_time,2),'초')

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test,y_predict,)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088

y_submit = model.predict(test_csv)  #test_csv를 pred 에(예측값에 넣고) y_서브밋에 저장
y_submit = np.argmax(y_submit, axis=1)

# loss: [0.24434563517570496, 0.9119750261306763]
# 걸린시간 : 53.95 초
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 663us/step
# acc_score: 0.911975