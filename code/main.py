import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import agent0 as A0
import agent1 as A1
import agent2 as A2

test = pd.read_csv('testing.csv', sep = ";")
test2 = pd.get_dummies(test.Tags, prefix = 'c')

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

for i in range(len(test2['c_Investment,Franchise,Stocks'])):
    if test2.iloc[i]['c_Investment,Franchise,Stocks'] == 1:
        test2.iloc[i]['c_Stocks'] = 1
        test2.iloc[i]['c_Franchise'] = 1
        test2.iloc[i]['c_Investment'] = 1
    elif test2.iloc[i]['c_Investment,Stocks'] == 1:
        test2.iloc[i]['c_Stocks'] = 1
        test2.iloc[i]['c_Investment'] = 1
    elif test2.iloc[i]['c_Franchise,Stocks'] == 1:
        test2.iloc[i]['c_Franchise'] = 1
        test2.iloc[i]['c_Stocks'] = 1
    elif test2.iloc[i]['c_Investment,Franchise'] == 1:
        test2.iloc[i]['c_Franchise'] = 1
        test2.iloc[i]['c_Investment'] = 1

test2 = test2.drop(['c_Investment,Franchise,Stocks', 'c_Investment,Stocks', 'c_Franchise,Stocks', 'c_Investment,Franchise'], axis=1)
test = test.drop(['Tags'], axis = 1)

X = test.iloc[:, 0:5]
Y = test.iloc[:, 11:12]
Ybuff = Y

ord_enc_1 = OrdinalEncoder()
ord_enc_2 = OrdinalEncoder()
X["Wordnum"] = ord_enc_1.fit_transform(X[["Word"]])
Y["Category"] = ord_enc_2.fit_transform(Y[["Decision"]])

X = X.iloc[:, 1:6]
Y = Y.iloc[:, 1:2]
X = pd.concat([test2, X], axis=1)

y_pred_0 = A0.model0.predict(X)
y_pred_0 = NormalizeData(y_pred_0)
y_pred_0 = list(y_pred_0)
y_pred_0 = edit(y_pred_0)

y_pred_0 = pd.DataFrame(y_pred_0, columns=["A0_pred"])
buff0 = y_pred_0
y_pred_0["A0PNum"] = ord_enc_2.fit_transform(y_pred_0)
y_pred_0 = y_pred_0.iloc[:, 1:2]
X = pd.concat([X, y_pred_0], axis = 1)

y_pred_1 = A1.model1.predict(X)
y_pred_1 = NormalizeData(y_pred_1)
y_pred_1 = list(y_pred_1)
y_pred_1 = edit(y_pred_1)

y_pred_1 = pd.DataFrame(y_pred_1, columns=["A1_pred"])
buff1 = y_pred_1
y_pred_1["A1PNum"] = ord_enc_2.fit_transform(y_pred_1)
y_pred_1 = y_pred_1.iloc[:, 1:2]
X = pd.concat([X, y_pred_1], axis = 1)

y_pred_2 = A2.model2.predict(X)
y_pred_2 = NormalizeData(y_pred_2)
y_pred_2 = list(y_pred_2)
y_pred_2 = edit(y_pred_2)
y_pred_2 = pd.DataFrame(y_pred_2, columns=["A2_pred"])

results = pd.DataFrame()
results = pd.concat([results, buff0], axis = 1)
results = pd.concat([results, buff1], axis = 1)
results = pd.concat([results, y_pred_2], axis = 1)
results = pd.concat([results, Ybuff], axis = 1)
results = results.drop(["A0PNum", "A1PNum", "Category"], axis = 1)
# print(results)
v0, v1, v2 = 0, 0, 0

for i in range(len(results)):
    if results.iloc[i]["A0_pred"] == results.iloc[i]["Decision"]:
        v0 += 1
    if results.iloc[i]["A1_pred"] == results.iloc[i]["Decision"]:
        v1 += 1
    if results.iloc[i]["A2_pred"] == results.iloc[i]["Decision"]:
        v2 += 1

print(v0, v1, v2)