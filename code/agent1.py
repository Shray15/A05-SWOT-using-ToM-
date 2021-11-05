import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import agent0 as A0

ord_enc1 = OrdinalEncoder()
ord_enc12 = OrdinalEncoder()

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


data00 = pd.read_csv('agent1train.csv', sep=";")
data11 = pd.get_dummies(data00.Tags, prefix = 'c')
for i in range(len(data11['c_Investment,Franchise,Stocks'])):
    if data11.iloc[i]['c_Investment,Franchise,Stocks'] == 1:
        data11.iloc[i]['c_Stocks'] = 1
        data11.iloc[i]['c_Franchise'] = 1
        data11.iloc[i]['c_Investment'] = 1
    elif data11.iloc[i]['c_Investment,Stocks'] == 1:
        data11.iloc[i]['c_Stocks'] = 1
        data11.iloc[i]['c_Investment'] = 1
    elif data11.iloc[i]['c_Franchise,Stocks'] == 1:
        data11.iloc[i]['c_Franchise'] = 1
        data11.iloc[i]['c_Stocks'] = 1
    elif data11.iloc[i]['c_Investment,Franchise'] == 1:
        data11.iloc[i]['c_Franchise'] = 1
        data11.iloc[i]['c_Investment'] = 1

data11 = data11.drop(['c_Investment,Franchise,Stocks', 'c_Investment,Stocks', 'c_Franchise,Stocks', 'c_Investment,Franchise'], axis=1)
data00 = data00.drop(['Tags'], axis = 1)

X00 = data00.iloc[:, 0:5]
Y00 = data00.iloc[:, 11:12]


X00["Wordnum"] = ord_enc1.fit_transform(X00[["Word"]])
Y00["Category"] = ord_enc12.fit_transform(Y00[["Decision"]])

X00 = X00.iloc[:, 1:6]
Y00 = Y00.iloc[:, 1:2]
X00 = pd.concat([data11, X00], axis=1)

y_pred00 = A0.model0.predict(X00)
y_pred00 = NormalizeData(y_pred00)
y_pred00 = list(y_pred00)
y_pred00 = edit(y_pred00)

y_pred00 = pd.DataFrame(y_pred00, columns=["A0_pred"])
y_pred00["A0PNum"] = ord_enc12.fit_transform(y_pred00)
y_pred00 = y_pred00.iloc[:, 1:2]
X00 = pd.concat([X00, y_pred00], axis = 1)

model1 = LinearRegression().fit(X00, Y00)
y_pred1 = model1.predict(X00)
y_pred1 = NormalizeData(y_pred1)
y_pred1 = list(y_pred1)
y_pred1 = edit(y_pred1)
y_pred1 = pd.DataFrame(y_pred1)

