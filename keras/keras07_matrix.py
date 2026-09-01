import numpy as np

x1 = np.array([1,2,3])                  #(3,)
print("x1 = ", x1.shape)

x2 = np.array([[1,2,3]])                #(1,3)
print("x2 = ", x2.shape)

x3 = np.array([[1,2],[3,4]])            #(2,2)
print("x3 = ", x3.shape)

x4 = np.array([[1,2],[3,4],[5,6,]])      #(3,2) 마지막 ,는 추가적인 데이터를 더 받을 수 있다는 의미
print("x4 = ", x4.shape)

#x5 = np.array([[1,2],[3,4],[5,6,7]])     #shape error
#print("x5 = ", x5.shape)

x5 = np.array([[[1,2],[3,4],[5,6]]])
print("x5 = ",x5.shape)

x6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print("x6 = ",x6.shape)

x7 = np.array([[[[[1,2,3,4,5],[6,7,8,9,10]]]]])
print("x7 = ",x7.shape)

x8 = np.array([[[1,2,3]],[[4,5,6]]])
print("x8 = ",x8.shape)

x9 = np.array([[[[1]]],[[[2]]]])
print("x9 = ",x9.shape)