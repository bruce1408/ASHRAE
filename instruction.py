import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import string
from operator import itemgetter
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib

##Read configuration parameters

train_file = "./data/train.csv"
MODEL_PATH = "./"
test_file = "./data/test.csv"
SUBMISSION_PATH = "./"
seed = 0


# print(train_file, seed)


# 清理和处理数据
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    # print(big_string)
    return np.nan


le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()


def clean_and_munge_data(df):
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

    df['HighLow'] = df['Pclass']
    df.loc[(df.Fare_Per_Person < 8), 'HighLow'] = 'Low'
    df.loc[(df.Fare_Per_Person >= 8), 'HighLow'] = 'High'

    le.fit(df['Sex'])
    x_sex = le.transform(df['Sex'])
    df['Sex'] = x_sex.astype(np.float)

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

    df = df.drop(['PassengerId', 'Name', 'Age'], axis=1)  # remove Name, Age and PassengerId

    return df


# 读取数据
traindf = pd.read_csv(train_file)
testdf = pd.read_csv(test_file)
##清洗数据
df = clean_and_munge_data(traindf)
df_test = clean_and_munge_data(testdf)
########################################formula################################

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'

y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
print(x_train)
print("the y train shape is: ", y_train.shape, x_train.shape)

##选择训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
# 初始化分类器
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
                             random_state=seed,
                             verbose=0)

###grid search找到最好的参数
param_grid = dict()
# ##创建分类pipeline
pipeline = Pipeline([('clf', clf)])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy',
                           cv=StratifiedShuffleSplit(test_size=0.2, random_state=seed)).fit(X_train, Y_train)
# 对结果打分
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)

print('-----grid search end------------')
print('on all train set')
scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=3, scoring='accuracy')
print(scores.mean(), scores)
print('on test set')
scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test, cv=3, scoring='accuracy')
print(scores.mean(), scores)

# 对结果打分

print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train)))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test)))

model_file = MODEL_PATH + 'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)

# import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
# X = np.array([[1, 2], [3, 4], [3, 2], [2, 4], [5, 2], [5, 4]])
# y = np.array([0, 0, 0, 1, 1, 1])
# sss = StratifiedShuffleSplit(n_splits=6, test_size=0.2, random_state=0)
# sss.get_n_splits(X, y)
#
# for train_index, test_index in sss.split(X, y):
#     print("TRAIN:\n", X[train_index])
#     print("label\n", y[train_index])
#     print("TEST:\n", X[test_index])
#     print("*"*5)

df_test["Survived"] = 0
print(df_test.info())

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'

y_test, x_test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
y_predict = grid_search.predict(x_test)
result = pd.DataFrame({"PassengerId": testdf['PassengerId'].values, "Survived": y_predict.astype(np.int32)})
result.to_csv('./result_gridCV.csv', index=False)
# print("the y train shape is: ", x_train.shape)
