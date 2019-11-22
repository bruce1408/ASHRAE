# coding=utf-8
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree
from sklearn.externals import joblib

# # set print mode
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)
#
# # age and cabin is fullless
# dataPath = "/home/bruce/PycharmProjects/ashrae/data"
# df = pd.read_csv("./data/train.csv")
# withOutName = df.drop(["Name", "Cabin", "Ticket"], axis=1)  # 去掉年龄和住的房间号码
# # print(withOutName.info())
# # print(withOutName.describe())
#
# # print(withOutName[withOutName.isnull().values==True].drop_duplicates())
# # print(withOutName[['Ticket', 'Pclass']].sort_values(by=['Pclass']))
# modeNum = withOutName["Embarked"].mode().iloc[0]
# print(modeNum)
# withOutName['Embarked'].fillna(modeNum, inplace=True)
# # print('fill na: ', withOutName.count())
#
# # 先预测年龄
# withOutName = df.drop(["Name", "Cabin", "Ticket"], axis=1)  # 去掉年龄和住的房间号码
# modeNum = withOutName["Embarked"].mode().iloc[0]  # 求众数
# withOutName['Embarked'].fillna(modeNum, inplace=True)  # 缺失值填补众数
# testData = withOutName[withOutName['Age'].isnull()]  # 年龄为空
# trainData = withOutName[withOutName['Age'].notnull()]  # 年龄不是空


def featureBuild(dataFrame, feature):

    ageDataSex = pd.get_dummies(dataFrame[['Sex']])
    ageDataEmb = pd.get_dummies(dataFrame[['Embarked']])
    ageData = dataFrame.join(ageDataSex)
    ageData = ageData.join(ageDataEmb)
    ageData = ageData.drop(["Sex", "Embarked"], axis=1)
    X = ageData[["Pclass", "Parch", "Fare", "SibSp", "Embarked_C", "Embarked_Q",
                 "Embarked_S", "Sex_male", "Sex_female"]]
    Y = ageData[feature]
    return X.values, Y.values


# X, Y = featureBuild(trainData, "Age")
# px, _ = featureBuild(testData, "Age")
# trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25, random_state=0)
# # reg = linear_model.Ridge(alpha=0.5)
# reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
# reg.fit(trainX, trainY)
# joblib.dump(reg, "./model/regression_model.m")
# y_predict = reg.predict(px)
# print(reg.score(testX, testY))

testPath = "/home/bruce/PycharmProjects/ashrae/data/test.csv"
df = pd.read_csv(testPath)
withOutName = df.drop(["Name", "Cabin", "Ticket"], axis=1)  # 去掉年龄和住的房间号码
modeNum = withOutName["Embarked"].mode().iloc[0]
print(modeNum)
withOutName['Embarked'].fillna(modeNum, inplace=True)
print(withOutName.describe())


reg = joblib.load("./model/regression_model.m")
# 先预测年龄
withOutName = df.drop(["Name", "Cabin", "Ticket"], axis=1)  # 去掉年龄和住的房间号码
modeNum = withOutName["Embarked"].mode().iloc[0]  # 求众数
withOutName['Embarked'].fillna(modeNum, inplace=True)  # 缺失值填补众数
testData = withOutName[withOutName['Age'].isnull()]  # 年龄为空
x, _ = featureBuild(testData, "Age")
y_predict = reg.predict(x)
for i in range(len(y_predict)):
    if y_predict[i] < 0:
        y_predict[i] = y_predict[i] * -1
testData["Age"] = y_predict

totalData = [withOutName[withOutName['Age'].notnull()], testData]
result = pd.concat(totalData)
result["Fare"].fillna(7.75, inplace=True)
print('the result info is: \n', result.info())

train_s, _ = featureBuild(result, "Age")
clf = joblib.load("./model/train_model.m")

predict_ = clf.predict(train_s)
print(predict_)
final = pd.read_csv("./data/gender_submission.csv")
final["Survived"] = predict_
final.to_csv("./data/result.csv", index=False)