import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import string
import matplotlib.pyplot as plt
from operator import itemgetter
import json
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib

pd.set_option('display.max_columns', None)
##Read configuration parameters

train_file = "./data/train.csv"
MODEL_PATH = "./"
test_file = "./data/test.csv"
SUBMISSION_PATH = "./"
seed = 0


def plot_NameLengthSurvived():
    """
    画图
    :return:
    """
    fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
    traindf['Name_length'] = traindf['Name'].apply(lambda x: len(x))
    name_length = traindf[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()
    # name_length.plot(kind='bar')
    sns.barplot(x='Name_length', y='Survived', data=name_length)  # 这个画法非常
    # plt.title("name length of survived")
    # plt.xlabel('name_length')
    # plt.ylabel('survived')
    plt.show()

# plot_NameLengthSurvived()


# 清理和处理数据
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    # print(big_string)
    return np.nan


le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()
scaler = preprocessing.StandardScaler()


def clean_and_munge_data(df, testPredict=False):
    # 处理缺省值
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)  # 价格如果缺省的话，直接用0来代替
    # 处理一下名字，生成Title字段
    # jonkheer 乡绅,capt 队长,don 西班牙语中的贵族,col 上校,major 少校,mlle 小姐,rev 牧师,done 女士,countess 女伯爵
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))  # 新增一列title，如果没有的话就用np.nan

    # 处理特殊的称呼，全处理成mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']

        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme', 'Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Miss']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title == '':
            if x['Sex'] == 'Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)

    # 看看家族是否够大，咳咳
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Family'] = df['SibSp'] * df['Parch']

    # 船票缺省的值用中位数来弥补
    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # 对缺失年龄用平均值来做预估
    df['AgeFill'] = df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[3]

    # 对年龄做一个分桶的操作
    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')  # 对缺失的登录港口用众数来代替

    df.loc[df.Cabin.isnull() == True, 'Cabin'] = 0.5  # Cabin是true的话,用0.5来代替
    df.loc[df.Cabin.isnull() == False, 'Cabin'] = 1.5

    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)

    # Age times class
    df['AgeClass'] = df['AgeFill'] * df['Pclass']
    df['ClassFare'] = df['Pclass'] * df['Fare_Per_Person']
    df['AgeSqure'] = df["AgeFill"].apply(lambda x: x ** 2)

    df['HighLow'] = df['Pclass']
    df.loc[(df.Fare_Per_Person < 8), 'HighLow'] = 'Low'
    df.loc[(df.Fare_Per_Person >= 8), 'HighLow'] = 'High'

    le.fit(df['Sex'])
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)

    # 对船票是否共享来新建一个特征
    dt = df["Ticket"].value_counts()
    data1 = pd.DataFrame(dt)
    share = data1[data1["Ticket"] > 1]
    share_ticket = share.index.values
    tickets = df.Ticket.values
    result = []
    for ticket in tickets:
        if ticket in share_ticket:
            ticket = 1
        else:
            ticket = 0
        result.append(ticket)
    results = pd.DataFrame(result)
    results.columns = ["Ticket_share"]
    df = pd.concat([df, results], axis=1)

    le.fit(df['Ticket'])
    x_Ticket = le.transform(df['Ticket'])
    df['Ticket'] = x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title = le.transform(df['Title'])
    df['Title'] = x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl = le.transform(df['HighLow'])
    df['HighLow'] = x_hl.astype(np.float)

    le.fit(df['AgeCat'])
    x_age = le.transform(df['AgeCat'])
    df['AgeCat'] = x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb = le.transform(df['Embarked'])
    df['Embarked'] = x_emb.astype(np.float)

    age_scale_param = scaler.fit(df[['AgeClass']])
    df['Age_scaled'] = scaler.fit_transform(df[['AgeClass']], age_scale_param)

    fare_scale_param = scaler.fit(df[['Fare_Per_Person']])
    df['Fare_Person_scaled'] = scaler.fit_transform(df[['Fare_Per_Person']], fare_scale_param)

    ticket_scale_param = scaler.fit(df[['Ticket']])
    df['Ticket_scaled'] = scaler.fit_transform(df[['Ticket']], ticket_scale_param)

    fare1_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare1_scale_param)

    fare_scale_param = scaler.fit(df[['ClassFare']])
    df['Class_scaled'] = scaler.fit_transform(df[['ClassFare']], fare_scale_param)

    age_scale_param = scaler.fit(df[['AgeSqure']])
    df['Age_suqre_scaled'] = scaler.fit_transform(df[['AgeSqure']], age_scale_param)
    df['Name_length'] = df["Name"].apply(len)

    # remove Name, Age and PassengerId
    # dropColumns = ['PassengerId', 'Name', 'Age', "ClassFare", 'Fare_Per_Person', 'AgeFill', 'Family', 'Ticket', 'Gender']
    # df = df.drop(dropColumns, axis=1)
    if testPredict:
        df = df[['Pclass', 'Title', 'Sex', 'AgeCat', 'Fare_Person_scaled', 'Fare', 'Family_Size', 'Age_suqre_scaled',
             'Ticket_scaled', 'Embarked', 'AgeClass']]
    else:
        df = df[['Survived', 'Pclass', 'Title', 'Sex', 'AgeCat', 'Fare_Person_scaled', 'Fare', 'Family_Size', 'Age_suqre_scaled',
             'Ticket_scaled', 'Embarked', 'AgeClass']]
    # print(df.corr())
    # print(df)
    return df


# 读取数据
traindf = pd.read_csv(train_file)
testdf = pd.read_csv(test_file)

# 清洗数据
df = clean_and_munge_data(traindf)
# print('the train data info is:\n', df.columns)
# #######################################formula################################
# formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Person_scaled+Fare+Family_Size+Age_suqre_scaled+Ticket_scaled' \
#              '+Embarked+AgeClass'
#
# y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
# print("the formula x train is: ", x_train.shape)
# print(x_train)
# y_train = np.asarray(y_train).ravel()
# print("the y train shape is {} the x_train shape is {}:".format(y_train.shape, x_train.shape))

y_train = df.values[:, 0].astype(np.float)
x_train = df.values[:, 1:]
print(x_train.shape)
y_train = np.asarray(y_train).ravel()
# print(y_train.shape)

# 选择训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
# 初始化分类器
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1,
                             random_state=seed,
                             verbose=0)

# grid search找到最好的参数
param_grid = dict()
# 创建分类pipeline
pipeline = Pipeline([('clf', clf)])
grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=3, scoring='accuracy',
                           cv=StratifiedShuffleSplit(test_size=0.2, random_state=seed)).fit(X_train, Y_train)


best_params = grid_search.best_estimator_.get_params()

# 对结果打分
def resultAnalyse():
    print("*" * 40)
    print("Best score: %0.3f" % grid_search.best_score_)
    print(grid_search.best_estimator_)
    print('-----grid search enddata_train------------')
    print('on all train set')
    scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=3, scoring='accuracy')
    print('the score mean {} and the socre is {}: '.format(scores.mean(), scores))
    print('on test set')
    scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test, cv=3, scoring='accuracy')
    print(scores.mean(), scores)

    # 对结果打分
    print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train)))
    print('test data')
    print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test)))


# resultAnalyse()

# 模型保存
def saveModel():
    model_file = MODEL_PATH + 'model-rf.pkl'
    joblib.dump(grid_search.best_estimator_, model_file)


# 测试数据
df_test = clean_and_munge_data(testdf, True)

def predictTest():
    # testdf["Survived"] = 0
    df_test = clean_and_munge_data(testdf, True)
    df_test['Survived'] = 0
    df_test.drop(["Survived"], inplace=True, axis=1)
    # print(df_test.describe())
    # formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Person_scaled+Fare_scaled+Family_Size'
    # y_test, x_test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
    best = grid_search.best_estimator_
    # y_predict = best.predict(x_test)
    y_predict = best.predict(df_test.values)
    result = pd.DataFrame({"PassengerId": testdf['PassengerId'].values, "Survived": y_predict.astype(np.int32)})
    result.to_csv('./result_gridCV_best.csv', index=False)

# predictTest()

# stacking
ntrain = x_train.shape[0]
ntest = df_test.values.shape[0]
print(ntest)
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)
        return self.clf.fit(x, y).feature_importances_


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
# y_train = X_train['Survived'].ravel()
# train = X_train.drop(['Survived'], axis=1)
# x_train = train.values  # Creates an array of the train data
# x_test = X_test.values  # Creats an array of the test data

# X_train = X_train.values
# Y_train = Y_train
X_test = df_test.values

print(X_test.shape)
print(x_train.shape)
print(y_train.shape)
# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, X_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, X_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, X_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, X_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, X_test)  # Support Vector Classifier

print("Training is complete")
print(gb_oof_train.shape)
# print(gb_oof_train.ravel())


def importFeature():
    rf_feature = rf.feature_importances(x_train, y_train)
    et_feature = et.feature_importances(x_train, y_train)
    ada_feature = ada.feature_importances(x_train, y_train)
    gb_feature = gb.feature_importances(x_train, y_train)
    return rf_feature, et_feature, ada_feature, gb_feature


cols = df.columns.values[1:]
# print(cols)
rf_feature, et_feature, ada_feature, gb_feature = importFeature()
feature_dataFrame = pd.DataFrame({"features": cols,
                                  "RandomForestFeatureImportance": rf_feature,
                                  "ExtraTreesFeatureImportance": et_feature,
                                  "AdaboostFeatureImportance": ada_feature,
                                  "GradientBoostFeatureImportance": gb_feature
                                  })
# Create the new column containing the average of values
feature_dataFrame['mean'] = feature_dataFrame.mean(axis=1)  # axis = 1 computes the mean row-wise
# Create the new column containing the average of values
# print(feature_dataFrame)

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })

# print('the et_oof data is: ', et_oof_train.shape)
x_train_c = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test_c = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
# print(x_test_c.shape)
# print(x_train_c.shape)
# print(y_train)

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators=2000,
 max_depth=4,
 min_child_weight=2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='binary:logistic',
 nthread=-1,
 scale_pos_weight=1).fit(x_train_c, y_train)
print(x_train_c.shape)
print(y_train.shape)
# df_test = clean_and_munge_data(testdf, True)
# df_test['Survived'] = 0
# df_test.drop(["Survived"], inplace=True, axis=1)
print(df_test.values.shape)
print(testdf["PassengerId"])
predictions = gbm.predict(x_test_c)

StackingSubmission = pd.DataFrame({'PassengerId': testdf['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)