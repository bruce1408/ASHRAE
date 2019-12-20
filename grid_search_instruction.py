from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

print("Size of training set:{} size of testing set:{}".format(X_train.shape[0], X_test.shape[0]))
print(X_train.shape)
# grid search start
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C, kernel='linear')  # 对于每种参数可能的组合，进行一次训练；
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:  # 找到表现最好的参数
            best_score = score
            best_parameters = {'gamma': gamma, 'C': C}

# grid search end
print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
print(svm.coef_)


# 数据集分3部分 train, validation, test
# X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=1)
# print("Size of training set:{} size of validation set:{} size of teseting set:{}".
#       format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
#
# best_score = 0.0
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         svm = SVC(gamma=gamma, C=C, kernel='linear')
#         svm.fit(X_train, y_train)
#         score = svm.score(X_val, y_val)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'gamma': gamma, 'C': C}
# svm = SVC(**best_parameters)  # 使用最佳参数，构建新的模型
# svm.fit(X_trainval, y_trainval)  # 使用训练集和验证集进行训练，more data always results in good performance.
# test_score = svm.score(X_test, y_test)  # evaluation模型评估
# print("Best score on validation set:{:.2f}".format(best_score))
# print("Best parameters:{}".format(best_parameters))
# print("Best score on test set:{:.2f}".format(test_score))


# gridSearch
# 把要调整的参数以及其候选值 列出来；
# param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100], "C": [0.001, 0.01, 0.1, 1, 10, 100]}
# print("Parameters:{}".format(param_grid))
#
# grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # 实例化一个GridSearchCV类
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=10)
# grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
# print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
# print("Best parameters:{}".format(grid_search.best_params_))
# print("Best score on train set:{:.2f}".format(grid_search.best_score_))

