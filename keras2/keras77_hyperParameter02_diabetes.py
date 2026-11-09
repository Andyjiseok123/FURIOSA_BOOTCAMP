import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(datasets)
# print(x.shape, y.shape) #(442,10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=3721)

#2. 모델 구성
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128,node2=64,node3=32,node4=16,node5=8, lr=0.001):
    inputs = Input(shape=(10,),name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1,activation='linear',name='outputs')(x)

    model=Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameter():
    batchs = [32,16,8,1,64]
    optimizers = ['adam','rmsprop','adadelta']
    dropouts = [0.2,0.3,0.4,0.5]
    activations = ['relu','elu','selu','linear']
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    node4 = [128,64,32,16]
    node5 = [128,64,32,16]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5
    }

hyperparameters = create_hyperparameter()
# print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)       # keras와 sci-kit learn이 달라서 한번 변환시켜줘야함

model = RandomizedSearchCV(keras_model,hyperparameters,
                           cv=3, random_state=4123, n_iter=10, verbose=1)

#3. 훈현
import time
start = time.perf_counter()
model.fit(x_train,y_train,epochs=100, verbose=1)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ",model.best_params_)

#4. 평가 예측
print('model score : ',model.score(x_test,y_test))
print('best score : ',model.best_score_)

print('걸린시간 : ', round(end-start,2),'초')