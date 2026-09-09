# acc = 0.95
import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
# print(datasets.data.shape)              # (178, 13)
# print(datasets.feature_names)           # ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data
y = datasets['target']
# print(np.unique(y,return_counts=True))          #(array([0, 1, 2]), array([59, 71, 48]))


##################### 원핫 1. to_categorical ######################
#####################    from tensorflow   #######################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)

##################### 원핫 2. pd.get_dummies ######################
#######################    from pandas   #########################
# y = pd.get_dummies(y,dtype=int)
# print(y)


####################### 원핫 3. sklearn.processing ######################
#######################    from sci-kit learn   #########################
# from sklearn.preprocessing import OneHotEncoder
# category = y.reshape(-1,1)
# print(category)
# encoder = OneHotEncoder().fit_transform(category)
# y = encoder.toarray()
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.85,
                                                    shuffle= True,
                                                    random_state=12431,
                                                    stratify=y)

#2. 모델 구성
model = Sequential()
model.add(Dense(13,activation='relu',input_dim=13))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 컴파일 푼련
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train,epochs = 1000, batch_size = 8,
          verbose=1,
          validation_split = 70/85,
          callbacks = [es]
          )
end_time = time.time()


#4. 평가 예측
result = model.evaluate(x_test,y_test)
print("loss = ", result[0])
print("acc = ", round(result[1],2))

y_predict = model.predict(x_test)

# print(y_predict)
# print(y_predict[0])
# print(np.argmax(y_predict[0]))
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_predict)
print(y_test)

accuracy_score = accuracy_score(y_test,y_predict)
print("acc_score : ", accuracy_score)
print("소요시간 : ", round(end_time-start_time,2),"초")