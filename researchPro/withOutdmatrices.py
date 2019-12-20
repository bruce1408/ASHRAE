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
import seaborn as sns
import warnings
from dataPipline import clean_and_munge_data
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
# Read configuration parameters

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


# 读取数据
traindf = pd.read_csv(train_file)
testdf = pd.read_csv(test_file)
# 清洗以及处理数据, 测试数据
df = clean_and_munge_data(traindf)
df_test = clean_and_munge_data(testdf, True)


def corrPlot(df):
    plt.figure(figsize=(20, 20))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.corr(), linewidths=0.1, annot=True, cmap="Blues", linecolor='white', square=True, vmax=True)
    plt.tight_layout()
    plt.savefig("./corr.png", bbox_inches='tight')
    plt.show()


def plotCorr():
    corrData = df.corr()
    corrData["Survived"].sort_values(ascending=False).plot(kind='bar')
    plt.show()


# print('the train data info is:\n', df.columns)
# #######################################formula################################
# formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Person_scaled+Fare+Family_Size+Age_suqre_scaled+Ticket_scaled' \
#              '+Embarked+AgeClass'

# y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
# print("the formula x train is: ", x_train.shape)
# y_train = np.asarray(y_train).ravel()

y_train = df.values[:, 0].astype(np.float)
x_train = df.values[:, 1:]
y_train = np.asarray(y_train).ravel()
# print(y_train.shape)


def trainModel():
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


trainModel()

# 模型保存
def saveModel(grid_search):
    model_file = MODEL_PATH + 'model-rf.pkl'
    joblib.dump(grid_search.best_estimator_, model_file)


def predictTest(grid_search):
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
ntrain = x_train.shape[0]  # 891
ntest = df_test.values.shape[0]  # 481
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))  # [5, 418]

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
    # 'max_features': 0.2,
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    # 'max_features': 0.5,
    'n_jobs': -1,
    'n_estimators': 500,
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
    # 'max_features': 0.2,
    'n_estimators': 500,
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
        # print(self.clf.fit(x, y).feature_importances_)
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
X_test = df_test.values  # (418, 11)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, X_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, X_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, X_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, X_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, X_test)  # Support Vector Classifier

print("Training is complete")
# print('gb train ', gb_oof_train.shape)  # [891, 1]
# print('gb test ', gb_oof_test.shape)  # [418, 1]


def importFeature():
    rf_feature = rf.feature_importances(x_train, y_train)
    et_feature = et.feature_importances(x_train, y_train)
    ada_feature = ada.feature_importances(x_train, y_train)
    gb_feature = gb.feature_importances(x_train, y_train)
    return rf_feature, et_feature, ada_feature, gb_feature


cols = df.columns.values[1:]
rf_feature, et_feature, ada_feature, gb_feature = importFeature()
feature_dataFrame = pd.DataFrame({"features": cols,
                                  "RandomForestFeatureImportance": rf_feature,
                                  "ExtraTreesFeatureImportance": et_feature,
                                  "AdaboostFeatureImportance": ada_feature,
                                  "GradientBoostFeatureImportance": gb_feature
                                  })
feature_dataFrame['mean'] = feature_dataFrame.mean(axis=1)  # axis = 1 computes the mean row-wise

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel()})

x_train_c = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)  # [891, 5]
x_test_c = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)  # [418, 5]


importanceNum = feature_dataFrame["RandomForestFeatureImportance"].sort_values(ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(feature_dataFrame['features'], feature_dataFrame["RandomForestFeatureImportance"].sort_values(ascending=True))
plt.savefig('./select_feature_selection.png')
plt.show()

gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
    # gamma=1,
    n_estimators=3000,
    max_depth=4,
    min_child_weight=2,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train_c, y_train)
# df_test = clean_and_munge_data(testdf, True)
# df_test['Survived'] = 0
# df_test.drop(["Survived"], inplace=True, axis=1)
predictions = gbm.predict(x_test_c)

StackingSubmission = pd.DataFrame({'PassengerId': testdf['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
