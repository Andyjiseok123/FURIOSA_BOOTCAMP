FURIOSA BOOT CAMP AI AGENT STUDY

1일차 (2026.08.31)   
&nbsp;&nbsp;&nbsp;&nbsp;-개발 환경 구축(python 설치, miniconda 가상환경 구축)   
&nbsp;&nbsp;&nbsp;&nbsp;-tensorflow 기본 모델 연습 (keras 기반)   
&nbsp;&nbsp;&nbsp;&nbsp;-Hyperparameter tuning(node, #of layer, epochs, batch size)   

2일차 (2026.09.01)   
&nbsp;&nbsp;&nbsp;&nbsp;-python을 이용한 matrix 기본 연산  
&nbsp;&nbsp;&nbsp;&nbsp;-Multi-layer percepticon 기본 구성  

3일차 (2026.09.02)   
&nbsp;&nbsp;&nbsp;&nbsp;-sci-kit learn 라이브러리를 이용한 data set 분리(train/test set)  
&nbsp;&nbsp;&nbsp;&nbsp;-sci-kit learn 라이브러리 내부의 훈련용 데이터를 활용한 학습  

4일차 (2026.09.03)   
&nbsp;&nbsp;&nbsp;&nbsp;-MSE, R2, RMSE와 같은 정확도, loss 지표 확인  
&nbsp;&nbsp;&nbsp;&nbsp;-def()를 활용한 python user function 생성  
&nbsp;&nbsp;&nbsp;&nbsp;-DAKON 데이터 활용 실습(따릉이 대여량 예측)  
&nbsp;&nbsp;&nbsp;&nbsp;-Pandas 모듈을 활용해 데이터 전처리  

5일차 (2026.09.04)  
&nbsp;&nbsp;&nbsp;&nbsp;-DAKON 데이터 활용 실습(따릉이 대여량 예측)  
&nbsp;&nbsp;&nbsp;&nbsp;-Kaggle 데이터 활용 실습(자전거 대여량 예측)  
&nbsp;&nbsp;&nbsp;&nbsp;-활성화 함수(Relu) 활용  

6일차 (2026.09.07)  
&nbsp;&nbsp;&nbsp;&nbsp;-verbose 옵션 활용  
&nbsp;&nbsp;&nbsp;&nbsp;-validation set 분리 - validation loss 도출 및 시각화  
&nbsp;&nbsp;&nbsp;&nbsp;-time 모듈을 활용한 학습 소요시간 계산  
&nbsp;&nbsp;&nbsp;&nbsp;-early stopping option을 통한 학습 최적화  

7일차 (2026.09.08)  
&nbsp;&nbsp;&nbsp;&nbsp;-이진 분류 (sigmoid, binary crossentropy)  

8일차 (2026.09.09)  
&nbsp;&nbsp;&nbsp;&nbsp;-다중 분류 (softmax, categorical crossentropy)  

|                                 | 회귀             | 이진 분류                | 다중 분류                     |
|:---:|:---:|:---:|:---:|
| One Hot Encoding               | X              | X                    | O                         |
| last layer Activation function | linear         | sigmoid              | softmax                   |
| last layer number of node      | N              | 1                    | number of class
| loss                           | MSE, MAE, .... | binary crossentropy | categorical crossentropy |
| predict                        | None           | np.round()           | np.argmax(,axis=)         |

9일차 (2026.09.10)  
&nbsp;&nbsp;&nbsp;&nbsp;-(결석) data scaling  

10일차 (2026.09.11)  
&nbsp;&nbsp;&nbsp;&nbsp;-(결석) save, load model/weights

11일차 (2026.09.14)  
&nbsp;&nbsp;&nbsp;&nbsp;-Model check point를 활용한 train 이력 저장  
&nbsp;&nbsp;&nbsp;&nbsp;-지금까지 사용한 dataset을 활용한 실습  
&nbsp;&nbsp;&nbsp;&nbsp;-dropout을 활용한 모델 성능 향상(과적합을 줄임)  
&nbsp;&nbsp;&nbsp;&nbsp;-funtional model을 활용한 기존 model 구현

12일차 (2026.09.15)  
&nbsp;&nbsp;&nbsp;&nbsp;-CUDA 환경 설정  

| Program | version |
|:---:|:---:|
| Nvidia graphic driver | 616.92 |
| cuda | 11.2 |
| cudnn | 8.1.1 for cuda 11.2 |
| 가상환경 | conda create -n tf29x-gpu python=3.10 |
| python | 3.10 |
| tensorflow | tensorflow-gpu==2.9.3 |
| numpy | uninstall and install 1.26.4 version |

13일차 (2026.09.16)  
&nbsp;&nbsp;&nbsp;&nbsp;-CNN 기초  

14일차 (2026.09.17)  
&nbsp;&nbsp;&nbsp;&nbsp;-conv2D에서 사용할 수 있는 option : padding, stride    
&nbsp;&nbsp;&nbsp;&nbsp;-Maxpooing2D와 GlobalAveragePooling2D  

15일차 (2026.09.18)  
&nbsp;&nbsp;&nbsp;&nbsp;-ImageDateGenerator : 기존의 이미지 데이터를 좌우반전, 상하반전등을 이용하여 데이터셋을 늘릴 수 있는 방법  
&nbsp;&nbsp;&nbsp;&nbsp;-np.save, np.load를 통한 이미지 데이터화(매번 변환을 통한 시간낭비 없앨 수 있음.)  

16일차 (2026.09.21)  
&nbsp;&nbsp;&nbsp;&nbsp;-레거시 ML 3대장 (XGboost, LGBM, Catboost)  
&nbsp;&nbsp;&nbsp;&nbsp;-종합적인 모델구성(save/load model, predict other picture)  
&nbsp;&nbsp;&nbsp;&nbsp;-기존 이미지 바탕으로 데이터 갯수 늘리기(ImageDataGenerator)  



