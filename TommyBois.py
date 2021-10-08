import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import math

def NormalizeData(data):
    scaler1 = MinMaxScaler(feature_range=(0, 4))
    return scaler1.fit_transform(data)


data0 = pd.read_csv('tom0.csv', sep=";")
data1 = pd.read_csv('tom1.csv', sep=";")
data2 = pd.read_csv('tom2.csv', sep=";")
test = pd.read_csv('testing.csv', sep=";")

X0 = data0.iloc[:, 0:5]
Y0 = data0.iloc[:, 11:12]
X1 = data1.iloc[:, 0:5]
Y1 = data1.iloc[:, 11:12]
X2 = data2.iloc[:, 0:5]
Y2 = data2.iloc[:, 11:12]
X3 = test.iloc[:, 0:5]
Y3 = test.iloc[:, 11:12]


ord_enc = OrdinalEncoder()
ord_enc2 = OrdinalEncoder()

X0["Wordnum"] = ord_enc.fit_transform(X0[["Word"]])
Y0["Category"] = ord_enc2.fit_transform(Y0[["Decision"]])
X0 = X0.iloc[:, 1:6]
Y0 = Y0.iloc[:, 1:2]

X1["Wordnum"] = ord_enc.fit_transform(X1[["Word"]])
Y1["Category"] = ord_enc2.fit_transform(Y1[["Decision"]])
X1 = X1.iloc[:, 1:6]
Y1 = Y1.iloc[:, 1:2]

X2["Wordnum"] = ord_enc.fit_transform(X2[["Word"]])
Y2["Category"] = ord_enc2.fit_transform(Y2[["Decision"]])
X2 = X2.iloc[:, 1:6]
Y2 = Y2.iloc[:, 1:2]

X3["Wordnum"] = ord_enc.fit_transform(X3[["Word"]])
Y3["Category"] = ord_enc2.fit_transform(Y3[["Decision"]])
X3 = X3.iloc[:, 1:6]
Y3 = Y3.iloc[:, 1:2]

model0 = LinearRegression().fit(X0, Y0)
y_pred0 = model0.predict(X1)
y_pred0 = NormalizeData(y_pred0)


X1 = np.append(X1, y_pred0, axis = 1)
X2 = np.append(X2, y_pred0, axis = 1)

model1 = LinearRegression().fit(X1, Y1)
y_pred1 = model1.predict(X2)
y_pred1 = NormalizeData(y_pred1)


X2 = np.append(X2, y_pred1, axis = 1)
X3 = np.append(X3, y_pred0, axis = 1)
X3 = np.append(X3, y_pred1, axis = 1)

model2 = LinearRegression().fit(X2, Y2)
y_pred2 = model2.predict(X3)
y_pred2 = NormalizeData(y_pred2)



y_pred0 = [int(x) for x in y_pred0]
y_pred1 = [int(x) for x in y_pred1]
y_pred2 = [int(x) for x in y_pred2]
print(y_pred0[1], y_pred1[1], y_pred2[1])