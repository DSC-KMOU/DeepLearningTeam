
import tensorflow as tf
import pandas as pd 
import numpy as np
#import matplotlib.contib as mlp


#file_root = r'C:\Users\ksi03\Downloads\RP_data\train\\'
file_root = r'C:\Users\ksi03\Downloads\RP_data\test\\'

token=".csv"
data_raw = pd.read_csv(file_root+'1547'+token, sep=',')

print("test 1547 shape: ")
print(data_raw.shape)



def is_real(file_path):
    data=pd.read_csv(file_path, sep=',')
    data=data.loc[0:1, :].values
    op_num=0

    ret="mimic"
    if data.dtype !='float64': ret="real"
    #5122개의 속성, 각 파일당 600개의 데이타, 


    return ret


#828개의 파일
real_num=0
for num in range(828, 1547):
    path=file_root+str(num)+token
    Real_or_Mimic=is_real(path)
    print(path+" is "+Real_or_Mimic)
    if Real_or_Mimic=="real": real_num+=1

print("real num= "+str(real_num))





'''
input_dim = len(dataset) #한 스텝당 정보의 수 를 자동으로 캡쳐
x_train = data_raw.loc[0:930000, dataset].values
x_train = x_train.reshape(x_train.shape[0], input_steps, input_dim) #형태 변환
y_train = data_raw.loc[0:930000, [label[i]]].values

#valid data 
x_valid = data_raw.loc[931001:981000,dataset].values
x_valid = x_valid.reshape(x_valid.shape[0], input_steps, input_dim) #형태 변환
y_valid = data_raw.loc[931001:981000, [label[i]]].values

#test data
x_test = data_raw.loc[981001:,dataset].values
x_test = x_test.reshape(x_test.shape[0], input_steps, input_dim) #형태 변환
y_test = data_raw.loc[981001:, [label[i]]].values
'''