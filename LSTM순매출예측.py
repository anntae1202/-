# lstm2 순매출

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pllt

raw_df=pd.read_csv("./Raw.csv",thousands = ',')
raw_df=raw_df.rename(columns={' 매출':'매출',' 순매출':'순매출',
                                ' 수량':'수량', ' 평균소비자가':'평균소비자가',
                                ' Stores':'Stores', ' SKU':'SKU'})

##### 새로운 변수 추가######

def weekday2(data):
    list=[]
    for i in data:
        # weekday()는 해당 날짜의 요일을 출력. 5는 토,6은 일요일
        if datetime.strptime(f"{i}", "%m/%d/%Y").weekday()>=5:
            list.append(1)
        else:
            list.append(0)
    return list

raw_df['week']=weekday2(raw_df['판매일자'])
# 주말 변수 : 주말 수량, 매출이 평일보다 많음.

# 높은 SKU 변수 : 24이상
highsku=(raw_df['SKU']>=24)
raw_df['highsku']=highsku.astype(int)


def MinMaxScaler1(data):
    # 정규화 코드
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

dfx = raw_df[['순매출','수량', '평균소비자가',
              '할인율', 'Stores', 'SKU',
              'week','highsku']] # 이거 갯수=data_size 수

dfx = MinMaxScaler1(dfx)
dfy = dfx[['순매출']]      

x = dfx.values.tolist()
y = dfy.values.tolist()



# 데이터셋 준비하기
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] 
    _y = y[i + window_size]     
    data_x.append(_x)
    data_y.append(_y)
print(_x, "->", _y)



# 훈련용데이터셋
train_size = int(len(data_y) * 0.7)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])



#테스트용 데이터셋
test_size = len(data_y) - train_size
test_x = np.array(data_x[train_size : len(data_x)])
test_y = np.array(data_y[train_size : len(data_y)])


window_size = 60 # 학습할 시계열 데이터수(60=60일)
data_size = len(dfx.columns) # 학습에 사용할 변수 갯수


# 모델 생성
model=Sequential()
model.add(LSTM(units=50,activation="relu",return_sequences=True,input_shape=(window_size, data_size)))
model.add(Dropout(0.2))
model.add(LSTM(units=60,activation="relu",return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80,activation="relu",return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=30)
pred_y = model.predict(test_x)


# print('##### Test Result #####')
# print('loss : ',str(loss_and_metrics[0]))
# print('Accuracy : ',str(loss_and_metrics[1]))

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print(f"RMSE 값 : {rmse}")
# sklearn은 mse만 지원하기 때문에 rmse와 rmsle는 직접 만들어야 합니다.
# y_preds는 예측값들이 담긴 데이터, y_test는 실제값 데이터들입니다.


# 역정규화 필요
scale=1/(dfx['순매출']/raw_df['순매출'])


#print(scale)
#%% 역정규화시켜야함.즉, 원래의 데이터로 만들어야해서
pred_y=pred_y*scale[0]
test_y=test_y*scale[0]
#print(pred_y)
#print(test_y)


# Visualising the results
%matplotlib inline

plt.figure()
pllt.rcParams["font.family"]="Malgun Gothic"
pllt.rcParams["font.size"]=12
pllt.rcParams["figure.figsize"]=(14,4)
mpl.rcParams['axes.unicode_minus']=False

plt.plot(test_y, color='red', label='실제 순매출')
plt.plot(pred_y, color='blue', label='예측 순매출')
plt.title('순매출 예측 그래프')
plt.xlabel('time')
plt.ylabel('순매출')
plt.legend()
plt.show()
%config InlineBackend.figure_format = 'retina'


b= pred_y[:30]
b