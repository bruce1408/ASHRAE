import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import string
import matplotlib.pyplot as plt
from operator import itemgetter
import json
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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
scaler = preprocessing.StandardScaler()


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

    df = df.drop(['PassengerId', 'Name', 'Age', "ClassFare", 'Fare_Per_Person'],
                 axis=1)  # remove Name, Age and PassengerId
    return df


# 读取数据
traindf = pd.read_csv(train_file)
testdf = pd.read_csv(test_file)

# 清洗数据
df = clean_and_munge_data(traindf)
# print(df.describe())

# #######################################formula################################

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Person_scaled+Fare+Family_Size+Age_suqre_scaled+Ticket_scaled' \
             '+Embarked+AgeClass+Name_length+Ticket_share'

y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
print('the train data\n', x_train.describe())
print("the y train shape is: ", y_train.shape, x_train.shape)

# 选择训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
# 初始化分类器
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
                             random_state=seed,
                             verbose=0)

# grid search找到最好的参数
param_grid = dict()
# 创建分类pipeline
pipeline = Pipeline([('clf', clf)])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy',
                           cv=StratifiedShuffleSplit(test_size=0.2, random_state=seed)).fit(X_train, Y_train)


# 对结果打分
def resultAnalyse():
    print("*" * 40)
    print("Best score: %0.3f" % grid_search.best_score_)
    print(grid_search.best_estimator_)

    print('-----grid search enddata_train------------')
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


resultAnalyse()
# 模型融合
def fill_missing_age(missing_age_train, missing_age_test):
    from sklearn import ensemble
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
    # 模型1
    gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [3], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(
        gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    # 模型2
    lrf_reg = LinearRegression()
    lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
    lrf_reg_grid = model_selection.GridSearchCV(lrf_reg, lrf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best LR Params:' + str(lrf_reg_grid.best_params_))
    print('Age feature Best LR Score:' + str(lrf_reg_grid.best_score_))
    print('LR Train Error for "Age" Feature Regressor' + str(
        lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_LRF'][:4])
    # 将两个模型预测后的均值作为最终预测结果
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_LRF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB','Age_LRF']].mode(axis=1)
    missing_age_test['Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_LRF']])
    print(missing_age_test['Age'][:4])
    drop_col_not_req(missing_age_test, ['Age_GB', 'Age_LRF'])

    return missing_age_test
model_file = MODEL_PATH + 'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)

df_test = clean_and_munge_data(testdf)
df_test["Survived"] = 0

# formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Person_scaled+Fare_scaled+Family_Size'
y_test, x_test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
best = grid_search.best_estimator_
y_predict = best.predict(x_test)
result = pd.DataFrame({"PassengerId": testdf['PassengerId'].values, "Survived": y_predict.astype(np.int32)})
result.to_csv('./result_gridCV_best.csv', index=False)


def plot_NameLengthSurvived():

    fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
    traindf['Name_length'] = traindf['Name'].apply(lambda x: len(x))
    name_length = traindf[['Name_length', 'Survived']].groupby(['Name_length'], as_index=False).mean()
    # name_length.plot(kind='bar')
    sns.barplot(x='Name_length', y='Survived', data=name_length)  # 这个画法非常
    # plt.title("name length of survived")
    # plt.xlabel('name_length')
    # plt.ylabel('survived')
    plt.show()


plot_NameLengthSurvived()