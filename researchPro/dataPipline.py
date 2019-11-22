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

train_file = "../data/train.csv"
MODEL_PATH = "./"
test_file = "../data/test.csv"
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


def clean_and_munge_data(df, Predict=False):
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
    df['Age'].fillna(method='ffill', inplace=True)
    df['Age'].fillna(method='bfill', inplace=True)
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')  # 对缺失的登录港口用众数来代替

    df.loc[df.Cabin.isnull()==True, 'Cabin'] = 0.5  # Cabin是true的话,用0.5来代替
    df.loc[df.Cabin.isnull()==False, 'Cabin'] = 1.5

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
    # dropColumns = ['PassengerId', 'Name', "ClassFare", 'Fare_Per_Person', 'AgeFill', 'Family', 'Ticket', 'Gender']
    # dropColumns = ['PassengerId', 'Name']
    # df = df.drop(dropColumns, axis=1)

    # the 0.79452
    # if Predict:
    #     df = df[['Pclass', 'Title', 'Sex', 'AgeCat', 'Fare_Person_scaled', 'Fare', 'Family_Size', 'Age_suqre_scaled',
    #          'Ticket_scaled', 'Embarked', 'AgeClass', 'Age']]
    # else:
    #     df = df[['Survived', 'Pclass', 'Title', 'Sex', 'AgeCat', 'Fare_Person_scaled', 'Fare', 'Family_Size',
    #     'Age_suqre_scaled', 'Ticket_scaled', 'Embarked', 'AgeClass', 'Age']]

    # the 0.70813
    # if Predict:
    #     df = df[['Name_length', 'Age_suqre_scaled', 'Class_scaled', 'Fare_scaled', 'Ticket_scaled', 'Fare_Person_scaled',
    #              'Age_scaled', 'Ticket_share', 'HighLow', 'AgeSqure', 'Fare_Per_Person', 'AgeCat']]
    # else:
    #     df = df[['Survived', 'Name_length', 'Age_suqre_scaled', 'Class_scaled', 'Fare_scaled', 'Ticket_scaled',
    #              'Fare_Person_scaled', 'Age_scaled', 'Ticket_share', 'HighLow', 'AgeSqure', 'Fare_Per_Person', 'AgeCat']]

    # the 0.77990
    # if Predict:
    #     df = df[['HighLow', 'Fare_scaled', 'Family_Size', 'Ticket_share', 'Sex',
    #              'Age_suqre_scaled', 'Ticket_scaled', 'Embarked', 'AgeClass', 'Age_scaled', 'Name_length',
    #              'Class_scaled']]
    # else:
    #     df = df[['Survived', 'HighLow', 'Fare_scaled', 'Family_Size', 'Ticket_share', 'Sex',
    #              'Age_suqre_scaled', 'Ticket_scaled', 'Embarked', 'AgeClass', 'Age_scaled', 'Name_length',
    #              'Class_scaled']]

    if Predict:
        df = df[['Sex', 'AgeCat', 'Fare_Person_scaled', 'Family_Size', 'Age_suqre_scaled',
             'Ticket_scaled', 'Embarked', 'AgeClass', 'Name_length', 'Class_scaled', 'Title', 'Fare',
                 'Pclass']]
    else:
        df = df[['Survived', 'Sex', 'AgeCat', 'Fare_Person_scaled', 'Family_Size',
        'Age_suqre_scaled', 'Ticket_scaled', 'Embarked', 'AgeClass', 'Name_length', 'Class_scaled', 'Title',
                 'Fare', 'Pclass']]

    return df


traindf = pd.read_csv(train_file)
testdf = pd.read_csv(test_file)

train = clean_and_munge_data(traindf)
test = clean_and_munge_data(testdf, Predict=True)
