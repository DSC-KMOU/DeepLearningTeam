import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

#x_train->일반 파일보다 약 84개의 column 제외됨
#y_train->0 or 1, 최초로 0이 나온 지점을 사용할 계획

#parameter
input_steps = 1 #입력 값의 총 스텝의 수
input_dim = 4001 #한 스텝당 정보의 수 (특징의 수 또는 피쳐의 수)
output_dim = 6 #출력 값의 정보의 수
size_input = input_steps * input_dim #입력 값의 총 정보량
dropout_rate = 0.15 #드롭아웃 비율
rnn_layers = 1 #순환 신경망(rnn) 층의 수
size_fc_layer = 100 #fully connected 연결망 층의 수
hidden_dim = 100 #hidden state의 크기
batch_size = 5 #배치 크기
start_time = time.time() #시작 시간
#


#"C:\Users\ksi03\Desktop\DeepLearningTeam\NuclearPowerPlantStatePrediction\이준혁\pred_time\pred_time\rawdata.csv"
file_root = r'C:\Users\ksi03\Downloads\RP_data\real_validated\\'
#data_raw=pd.read_csv(root, sep=',')

##데이터 합치기
#ind=['30', '1168', '1154', 'additional_data_01']
#ind=['1', '30','additional_data_01']
ind=['1', '30', '1168', '1154', 'additional_data_01', 'additional_data_02']
token=".csv"
data_raw=0
for i in range(len(ind)):
    tmp=pd.read_csv(file_root+ind[i]+token, sep=',')
    if i!=0: data_raw=pd.concat([data_raw, tmp], ignore_index=True)
    else: data_raw=tmp


##데이터 전처리
d1=data_raw
d1.dropna(axis=1)
for j in list(d1.columns[:]):
    if d1[j].dtype=='O' or d1[j].isnull().sum()!=0:
        del d1[j]
        #d1.drop([j], axis=1)

print("처리된 데이터타입: ")

print(d1.values.dtype)

#d1.to_csv("d1.csv")

##
##정규화##
print(d1.shape)
y_raw=d1[['label_0','label_1']]

del d1['label_0']
del d1['label_1']


#tempdata=d1.values
rowsize=d1.shape[0]
train_size=int(rowsize*0.5)
dataset=d1.columns[:]  

time_min=[]
time_max=[]
labels_processed=[]
for prop in list(dataset) :
    min_test = d1.loc[0:train_size, prop].min()
    max_test = d1.loc[0:train_size, prop].max()
    labels_processed.append(prop)
    time_min.append(min_test)
    time_max.append(max_test)
    if max_test-min_test!=0:
        d1[prop] = (d1[prop] - min_test) / (max_test-min_test)
        
        
print(d1)


#최종확인
print(d1.isnull().values.any())
#########
#dataset
#data_raw.shape=[1300,4010]
#train->data_raw의 90%
#valid->data_raw의 10%

x_train=d1.loc[0:train_size, :].values
x_valid=d1.loc[train_size:, :].values

y_train=y_raw.loc[0:train_size].values
y_valid=y_raw.loc[train_size:].values

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_valid)


#형변환
input_dim=x_train.shape[1]
x_train = x_train.reshape(x_train.shape[0], input_steps, input_dim) #형태 변환
x_valid = x_valid.reshape(x_valid.shape[0], input_steps, input_dim) #형태 변환
print("=========")
print(x_train.shape)
print(x_valid.shape)

#########

#러닝
##########
##모델
init=tf.keras.initializers.he_uniform()
reg=tf.keras.regularizers.l2(l=0.001)
droprate = 0.2
#act=tf.nn.leaky_relu
act='relu'

model = tf.keras.Sequential(
[
#tf.keras.layers.Input(shape=(input_dim)),
tf.keras.layers.Input(shape=(input_steps, input_dim)),
tf.keras.layers.GRU(hidden_dim ,return_state=False),
tf.keras.layers.Dense(500, activation=act),
#tf.keras.layers.Dropout(droprate),
tf.keras.layers.Dense(200, activation=act),
#tf.keras.layers.Dropout(droprate),
#tf.keras.layers.GaussianNoise(0.01),
tf.keras.layers.Dense(2, activation='softmax')])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])

##
def variable_extracter(hist):
    loss = hist.history['loss'][0]
    acc=hist.history['accuracy'][0]

    return loss, acc
##러닝
current_cost_train = 1000 #초기 값으로 높은 값 부여
current_cost_valid = 1000 #초기 값으로 높은 값 부여
known_best = 1000000 #초기 값으로 높은 값 부여
full_step = 11
best_model=0
trainValues=[]
validValues=[]
for step in range(62): #3000번의 학습 후 종료

    
    train_hist = model.fit(x_train, y_train, batch_size=batch_size, verbose=1)
    loss, acc = variable_extracter(train_hist)
    
    if (step > 0 and step % 10 == 0): #매 50번째 학습 성능 값 출력
        print("[step: {}]'s cost = {}   acc={}".format(step, loss, acc)) 
        if (step >= 0): #1000번째부터 검증데이터로 모델 성능 평가
            current_cost_valid = model.evaluate(x_valid, y_valid, verbose=0)

            trainValues.append(loss) #학습 곡선을 출력하기 위해 현재 값 저장
            validValues.append(current_cost_valid[0]) #검증 곡선을 출력하기위해 현재 값 저장

            if (current_cost_valid[0] < known_best): #검증 성능이 이전보다 좋을 경우 현재 모델 저장
                known_best = current_cost_valid[0]
                best_model=model
                print("New best known cost = {}".format(known_best))
    
print("학습루틴 {}번 스텝 종료".format(step))
root = 'Saved_models/rnn_model'+ '.h5'
save_path = best_model.save(root)
##
print("--- 학습 종료 ---")
print("--- %s seconds ---" % (time.time() - start_time))

########## 실 데이터로 확인

#모델 임포트
root = 'Saved_models/rnn_model'+ '.h5'
time_model=tf.keras.models.load_model(root)
#

#데이터 스케일링 함수, labels_processed 레이블 기준
def datascaler(dframe, labels_processed):
    dframe=dframe[labels_processed]
    return dframe
#

#데이터 특성치 로드 함수.
def load_optns(file_num_sum, column_arr, reffile_num=50):
    min_arr=[]
    max_arr=[]
    tmp_arr=[]

    file_root = r'C:\Users\ksi03\Downloads\RP_data\train\\'
    token='.csv'
    
    rand=random.sample(range(0, file_num_sum), reffile_num)
    
    k=True
    p=1
    for i in rand:
        
        print(str(p)+'번 파일')
        root=file_root+str(i)+token

        tmp_check=pd.read_csv(root, sep=',', nrows = 1)
        p=p+1
        if tmp_check.values.dtype!='O':
            tmp=pd.read_csv(root, sep=',')
            #concat all data
            if k!=True: tmp_arr=pd.concat([tmp_arr, tmp], ignore_index=True)
            else: 
                tmp_arr=tmp
                k=False
        

    for col in column_arr:
        if True:
            min_arr.append(tmp_arr.loc[:, col].min())
            max_arr.append(tmp_arr.loc[:,col].max())

    return min_arr, max_arr
    
#

#데이터 임포트 함수, 파일이 많기 때문에 항상 처리가 필요
def import_data(fileno):
    file_root = r'C:\Users\ksi03\Downloads\RP_data\train\\'
    token='.csv'
    root=file_root+str(fileno)+token
    tmp=pd.read_csv(root, sep=',', nrows=30)
    return tmp
#

#배치 제작 함수-로드, 스케일링, 리셰이핑까지 
#마지막 20개는 valid, 7개는 test
y_tmp=pd.read_csv(r"C:\\Users\ksi03\Downloads\RP_data\train_label.csv", sep=',')

def batch_make(n, min, max, size, model=time_model, column_scale=labels_processed):
    #start by 0

    #import y label
    
    data_raw=[]
    y=[]

    for i in range(n*size, (n+1)*size-1):
        i=5
        print('{}번 파일'.format(i))
        y_i=y_tmp.loc[n*size:(n+1)*size-1, 'label'][i]
        x_tmp=import_data(i)
        if x_tmp.values.dtype!='O': 
            #scaling by time options
            x_time_tmp=x_tmp[column_scale]
            
            for col in x_time_tmp.columns[:]:
                ord=x_time_tmp.columns[:].tolist().index(col)
                #mean_i=mean[ord]
                #std_i=std[ord]
                min_i=time_min[ord]
                max_i=time_max[ord]

                if max_i-min_i!=0:
                    x_tmp[col]=(x_tmp[col]-min_i)/(max_i-min_i)
            
            #predict y timing
            
            x_time_tmp=x_time_tmp.values.reshape(x_time_tmp.shape[0], input_steps, input_dim)
            #x_time_tmp=x_time_tmp.values
            y_pred_tmp=model.predict(x_time_tmp)
            y_pred=[]#0이면 a, 1이면 b
            for _ in range(y_pred_tmp.shape[0]):
                if y_pred_tmp[_, 0]>y_pred_tmp[_, 1]: y_pred.append(0)
                else: y_pred.append(1)
            loc=10
            loc=list(y_pred).index(1)
            '''
            for _ in range(y_pred):
                if y_pred[_, 1]==1:
                    print("yloc={}".format(_))
                    loc=_
                    break
            '''
            #make y
            
            for _ in range(loc):
                y.append(999)
            for _ in range(x_tmp.shape[1]-loc-1):
                y.append(y_i)


            #concat x
            if len(data_raw)!=0: data_raw=pd.concat([data_raw, x_tmp], ignore_index=True)
            else: data_raw=x_tmp
        else: print('real data '+str(i)+'번 파일 제외')



    return x, y
#

#데이터 임포트 및 스케일링, 리셰이핑
#만약 std가 0이라면 스케일링 진행 x

print('특성 임포팅 시작')
column_label=import_data(0).columns[:]
std, mean=load_optns(827, column_label, reffile_num=2)
print('특성 임포팅 성공')

print('첫 배치 임포트')
x_batch, y_batch=batch_make(0, std, mean, 50)
print("x={}".format(x_batch.shape))
print("y={}".format(y_batch.shape))
print('첫 배치 임포트 성공')
#

#
#예측, 첫 1이 나온 것 기준 데이터(변화시) 집계(파일 1개 당 변화시 하나)

#

#신규 모델 생성 및 학습(95프로() 트레인, 5프로(나머지) 배리드)

#

#submission 양식대로 변환

##########

