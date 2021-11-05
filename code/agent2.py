import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import agent0 as A0
import agent1 as A1

ord_enc2 = OrdinalEncoder()
ord_enc22 = OrdinalEncoder()

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


data000 = pd.read_csv('agent2train.csv', sep=";")
data111 = pd.get_dummies(data000.Tags, prefix = 'c')
for i in range(len(data111['c_Investment,Franchise,Stocks'])):
    if data111.iloc[i]['c_Investment,Franchise,Stocks'] == 1:
        data111.iloc[i]['c_Stocks'] = 1
        data111.iloc[i]['c_Franchise'] = 1
        data111.iloc[i]['c_Investment'] = 1
    elif data111.iloc[i]['c_Investment,Stocks'] == 1:
        data111.iloc[i]['c_Stocks'] = 1
        data111.iloc[i]['c_Investment'] = 1
    elif data111.iloc[i]['c_Franchise,Stocks'] == 1:
        data111.iloc[i]['c_Franchise'] = 1
        data111.iloc[i]['c_Stocks'] = 1
    elif data111.iloc[i]['c_Investment,Franchise'] == 1:
        data111.iloc[i]['c_Franchise'] = 1
        data111.iloc[i]['c_Investment'] = 1

data111 = data111.drop(['c_Investment,Franchise,Stocks', 'c_Investment,Stocks', 'c_Franchise,Stocks', 'c_Investment,Franchise'], axis=1)
data000 = data000.drop(['Tags'], axis = 1)

X000 = data000.iloc[:, 0:5]
Y000 = data000.iloc[:, 11:12]


X000["Wordnum"] = ord_enc2.fit_transform(X000[["Word"]])
Y000["Category"] = ord_enc22.fit_transform(Y000[["Decision"]])

X000 = X000.iloc[:, 1:6]
Y000 = Y000.iloc[:, 1:2]
X000 = pd.concat([data111, X000], axis=1)

y_pred000 = A0.model0.predict(X000)
y_pred000 = NormalizeData(y_pred000)
y_pred000 = list(y_pred000)
y_pred000 = edit(y_pred000)

y_pred000 = pd.DataFrame(y_pred000, columns=["A0_pred"])
y_pred000["A0PNum"] = ord_enc22.fit_transform(y_pred000)
y_pred000 = y_pred000.iloc[:, 1:2]
X000 = pd.concat([X000, y_pred000], axis = 1)

y_pred111 = A1.model1.predict(X000)
y_pred111 = NormalizeData(y_pred111)
y_pred111 = list(y_pred111)
y_pred111 = edit(y_pred111)

y_pred111 = pd.DataFrame(y_pred111, columns=["A1_pred"])
y_pred111["A1PNum"] = ord_enc22.fit_transform(y_pred111)
y_pred111 = y_pred111.iloc[:, 1:2]
X000 = pd.concat([X000, y_pred111], axis = 1)

model2 = LinearRegression().fit(X000, Y000)
y_pred2 = model2.predict(X000)
y_pred2 = NormalizeData(y_pred2)
y_pred2 = list(y_pred2)
y_pred2 = edit(y_pred2)
