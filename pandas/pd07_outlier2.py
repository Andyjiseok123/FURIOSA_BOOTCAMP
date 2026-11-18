import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,450]]).T

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data,[25,50,75])
    print('1사분위 : ',quartile_1)
    print('q2 : ',q2)
    print('3사분위 : ',quartile_3)
    iqr = quartile_3 - quartile_1
    print('IQR : ',iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data>upper_bound)|(data<lower_bound)),\
            iqr,lower_bound,upper_bound

# # 데이터 슬라이싱
# # column 1
# aaa1 = aaa[:,0].astype(float)
# print(aaa1)
# outlier_loc1, iqr, low, up = outlier(aaa1)
# print('이상치의 위치 : ',outlier_loc1)

# # 이상치를 결측치로 치환
# for i in outlier_loc1:
#     aaa1[i]=np.nan

# # column 2
# aaa2 = aaa[:,1].astype(float)
# print(aaa2)
# outlier_loc2, iqr, low, up = outlier(aaa2)
# print('이상치의 위치 : ',outlier_loc2)

# for i in outlier_loc2:
#     aaa2[i]=np.nan

# # 다시 합치기
# aaa = [aaa1,aaa2]
# print(np.array(aaa).T)


outlier_loc, iqr, low, up = outlier(aaa)
aaa_clipped = np.clip(aaa,low,up)
print(aaa_clipped)