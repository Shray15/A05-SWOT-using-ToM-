import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

ord_enc = OrdinalEncoder()
ord_enc2 = OrdinalEncoder()

def NormalizeData(data):
    scaler1 = MinMaxScaler(feature_range=(0, 3))
    k = scaler1.fit_transform(data)
    return k

def edit(data):
    for i in range(len(data)):
        if int(data[i]) == 1 :
            data[i] = 'S'
        elif int(data[i]) == 2 :
            data[i] = 'T'
        elif int(data[i]) == 3 :
            data[i] = 'W'
        elif int(data[i]) == 0 :
            data[i] = 'O'

    return data
            


data0 = pd.read_csv('agent0train.csv', sep=";")
data1 = pd.get_dummies(data0.Tags, prefix = 'c')
for i in range(len(data1['c_Investment,Franchise,Stocks'])):
    if data1.iloc[i]['c_Investment,Franchise,Stocks'] == 1:
        data1.iloc[i]['c_Stocks'] = 1
        data1.iloc[i]['c_Franchise'] = 1
        data1.iloc[i]['c_Investment'] = 1
    elif data1.iloc[i]['c_Investment,Stocks'] == 1:
        data1.iloc[i]['c_Stocks'] = 1
        data1.iloc[i]['c_Investment'] = 1
    elif data1.iloc[i]['c_Franchise,Stocks'] == 1:
        data1.iloc[i]['c_Franchise'] = 1
        data1.iloc[i]['c_Stocks'] = 1
    elif data1.iloc[i]['c_Investment,Franchise'] == 1:
        data1.iloc[i]['c_Franchise'] = 1
        data1.iloc[i]['c_Investment'] = 1

data1 = data1.drop(['c_Investment,Franchise,Stocks', 'c_Investment,Stocks', 'c_Franchise,Stocks', 'c_Investment,Franchise'], axis=1)
data0 = data0.drop(['Tags'], axis = 1)

X0 = data0.iloc[:, 0:5]
Y0 = data0.iloc[:, 11:12]


X0["Wordnum"] = ord_enc.fit_transform(X0[["Word"]])
Y0["Category"] = ord_enc2.fit_transform(Y0[["Decision"]])

X0 = X0.iloc[:, 1:6]
Y0 = Y0.iloc[:, 1:2]

X0 = pd.concat([data1, X0], axis=1)

model0 = LinearRegression().fit(X0, Y0)
y_pred0 = model0.predict(X0)
y_pred0 = NormalizeData(y_pred0)
y_pred0 = list(y_pred0)
y_pred0 = edit(y_pred0)
y_pred0 = pd.DataFrame(y_pred0)