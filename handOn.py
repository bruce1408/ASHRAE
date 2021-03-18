# coding=utf-8
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# age and cabin is fullless
dataPath = "/home/bruce/PycharmProjects/ashrae/data"
df = pd.read_csv("./data/train.csv")
withOutName = df.drop(["Name", "Cabin", "Ticket"], axis=1)  # 去掉年龄和住的房间号码
print(withOutName.info())
print(withOutName.describe())

print(withOutName[withOutName.isnull().values == True].drop_duplicates())
# print(withOutName[['Ticket', 'Pclass']].sort_values(by=['Pclass']))
withOutName['Embarked'].fillna('S', inplace=True)
print(withOutName.info())

# 先预测年龄
agePredictData = withOutName[withOutName['Age'].isnull()]
ageData = withOutName[withOutName['Age'].notnull()]

def extractInfo(ageData):

    print(ageData.info())
    ageDataSex = pd.get_dummies(ageData[['Sex']])
    ageDataEmb = pd.get_dummies(ageData[['Embarked']])
    ageData = ageData.join(ageDataSex)
    ageData = ageData.join(ageDataEmb)
    ageData = ageData.drop(["Sex", "Embarked"], axis=1)
    print(ageData.info())
    X = ageData[["Pclass", "Parch", "Fare", "SibSp", "Embarked_C", "Embarked_Q", "Embarked_S", "Sex_male", "Sex_female"]]
    X = ageData[["Pclass", "Parch", "Fare", "Survived", "SibSp", "Embarked_C", "Embarked_Q", "Embarked_S",
                 "Sex_male", "Sex_female"]]
    Y = ageData["Survived"]
    return X, Y


X, Y = extractInfo(ageData)
px, py = extractInfo(agePredictData)
print('the X is: ', X)
# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(X['Age'])
# print(px)
trainX, testX, trainY, testY = train_test_split(X.values, Y.values, test_size=0.25, random_state=0)
# reg = linear_model.Ridge(alpha=0.5)# print(result.info())
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit(trainX, trainY)
y_predict = reg.predict(px)
print(y_predict)
for i in range(len(y_predict)):
    if y_predict[i] < 0:
        y_predict[i] = y_predict[i] * -1
# print(y_predict)
agePredictData["Age"] = y_predict
totalData = [agePredictData, ageData]
result = pd.concat(totalData)

x, y = extractInfo(result)
# print("the x des is: ", x.describe())
# print(y.values)
train_x, test_x, train_y, test_y = train_test_split(x.values, y.values, test_size=0.1)
clf = tree.DecisionTreeClassifier()
clf.fit(train_x, train_y)
joblib.dump(clf, "./model/train_model_v1.m")
score = clf.score(test_x, test_y)
# print("the result is: ", score)

# import sklearn.preprocessing as preprocessing
# 测试




