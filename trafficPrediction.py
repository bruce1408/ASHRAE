# coding: utf-8
# 交调站车流量预测分析

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
pd.set_option('display.max.columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def loadData(originDataPath):
    """
    例如： originDataPath: origin.csv 文件
    导入原始数据，导入之前需要预处理；
    将原始数据中的带有单位的 米、% 号去除；
    :param originDataPath:原始数据路径
    :return:
    """
    totalPath = originDataPath
    df = pd.read_csv(totalPath, header=0, names=['Devices', 'Lane', 'PercentageCar', 'AverageDistance', 'TimeOccupy', 'Traffic', 'Speed', 'Time'])
    devicesName = df['Devices'].unique()
    print('the devices list is: ', devicesName)
    for eachName in devicesName:
        df[df["Devices"] == eachName].to_csv(str(eachName)+"_device.csv", index=False)
    return devicesName


# devicesName = [51121312100001, 141150311111133, 141150311111128, 171170318052615,
#             171170318052613, 171170318052614, 171170318052611,  51121311060001,
#             171170317010013, 141150315060501]


def drawPlot(y_predict, y_test, lane, index):
    """
    画图，输入预测结果，真实数据，车道号，设备号
    :param y_predict: 模型预测结果
    :param y_test: 数据测试集真实数据
    :param index: 设备编号
    :param lane: 车道
    :return: 保存图片
    """
    fig, axis1 = plt.subplots(1, 1, figsize=(20, 4))
    fig = plt.figure(num=1, figsize=(20, 8), dpi=80)

    plt.subplot(2, 1, 1)
    plt.plot(y_test[0:100], 'g-', linewidth=2, alpha=0.75)
    plt.ylabel("Traffic", fontsize=15)
    plt.title("True data", fontsize=20)

    plt.subplot(2, 1, 2)
    plt.plot(y_predict[0:100], 'r-', linewidth=3, alpha=0.6)
    plt.ylabel("Traffic", fontsize=15)
    plt.xlabel("Time", fontsize=15)
    plt.title("Prediction data", fontsize=20)
    plt.savefig("random_" + str(index) + "_" + str(lane)+".png")
    # print(index)
    plt.clf()


def buildOneDevices(devicesNum):
    """
    预测单个设备的结果，例如：devicesNum=171170318052615，
    :param devicesNum: 设备号
    :return:
    """
    dv1 = pd.read_csv(str(devicesNum)+"_device.csv")
    # 得到该设备号的多个车道list
    laneNum = dv1['Lane'].unique()
    print("设备 %d 车道号: " % devicesNum, laneNum)
    dv1['Time'] = pd.to_datetime(dv1['Time'])

    # 对一个设备的多个车道号进行预测
    # 数据处理以及特征构建

    for lane in laneNum:
        dv1Upstream = dv1[dv1['Lane'] == lane]
        # 按照时间进行排序
        dv1.sort_values('Time', inplace=True)
        dv1Upstream.sort_values('Time', inplace=True)
        dv1Upstream.reset_index(drop=True, inplace=True)

        temp = pd.DatetimeIndex(dv1Upstream['Time'])
        dv1Upstream['date'] = temp.date
        dv1Upstream['time'] = temp.time

        dv1Upstream['dayofweek'] = pd.DatetimeIndex(dv1Upstream.date).dayofweek
        dv1Upstream.drop(["Time"], axis=1, inplace=True)

        dv1Upstream['hour'] = pd.to_datetime(dv1Upstream.time, format="%H:%M:%S")
        dv1Upstream["hour"] = pd.Index(dv1Upstream['hour']).hour

        # 一周星期几对于流量分布的影响 ,0是周一
        dayWeek = dv1Upstream.groupby('dayofweek')
        dayWeek['Traffic'].sum().reset_index()

        # 对于异常特征用以下的这条特征来做
        # data_train = dv1Upstream.drop(["Devices", "Lane", "date", "time", "PercentageCar",
        #                                "AverageDistance", "TimeOccupy"])
        data_train = dv1Upstream.drop(['Devices', 'Lane', 'date', 'time'], axis=1)

        # speed 使用上一个值或者是下一个值进行填充
        data_train['Speed'].fillna(method='pad', inplace=True)
        data_train["Speed"].fillna(method='bfill', inplace=True)

        y = data_train.values[:, 0]
        x = data_train.values[:, 1:]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        # 构建模型，开始训练
        rfr = RandomForestRegressor(random_state=0, n_estimators=3000, n_jobs=-1)
        rfr.fit(x_train, y_train)
        rfr.score(x_test, y_test)
        y_predict = rfr.predict(x_test)
        # 结果指标分析
        print("========== 结果分析 =========")
        print("the mse is: ", mean_squared_error(y_test, y_predict))
        print("the mae is: ", mean_absolute_error(y_test, y_predict))
        print("the r2 is: ", r2_score(y_test, y_predict))
        print("="*28)
        # 画图保存
        drawPlot(y_predict, y_test, lane, devicesNum)


def buildModel(devicesName):
    """
    输入是从原始数据集中按照设备号来进行预测；默认选择的是 11 车道；
    :param devicesName: 设备号列表；
    :return:
    """

    for i in devicesName:
        fig = plt.subplots(figsize=(20, 8), facecolor="white")
        # fig = plt.figure(num=1, figsize=(20, 8), dpi=80)
        dv1Path = str(i) + "_device.csv"

        dv1 = pd.read_csv(dv1Path)
        print("设备 %d 车道 : "%i, dv1['Lane'].unique())  # 11, 31
        dv1['Time'] = pd.to_datetime(dv1['Time'])
        dv1Upstream = dv1[dv1['Lane'] == 11]

        # 按照时间进行排序
        dv1.sort_values('Time', inplace=True)
        dv1Upstream.sort_values('Time', inplace=True)
        dv1Upstream.reset_index(drop=True, inplace=True)

        temp = pd.DatetimeIndex(dv1Upstream['Time'])
        dv1Upstream['date'] = temp.date
        dv1Upstream['time'] = temp.time

        dv1Upstream['dayofweek'] = pd.DatetimeIndex(dv1Upstream.date).dayofweek
        dv1Upstream.drop(["Time"], axis=1, inplace=True)

        dv1Upstream['hour'] = pd.to_datetime(dv1Upstream.time, format="%H:%M:%S")
        dv1Upstream["hour"] = pd.Index(dv1Upstream['hour']).hour

        # 一周星期几对于流量分布的影响 ,0是周一
        dayWeek = dv1Upstream.groupby('dayofweek')
        dayWeek['Traffic'].sum().reset_index()

        data_train = dv1Upstream.drop(['Devices', 'Lane', 'date', 'time'], axis=1)

        # speed 使用上一个值进行填充
        data_train['Speed'].fillna(method='pad', inplace=True)
        data_train["Speed"].fillna(method='bfill', inplace=True)
        # print(data_train.describe())

        y = data_train.values[:, 0]
        x = data_train.values[:, 1:]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        rfr = RandomForestRegressor(random_state=0, n_estimators=3000, n_jobs=-1)
        rfr.fit(x_train, y_train)
        rfr.score(x_test, y_test)
        y_predict = rfr.predict(x_test)
        print("========== 结果分析 =========")
        print('the mse is: ', mean_squared_error(y_test, y_predict))
        print("the mae is: ", mean_absolute_error(y_test, y_predict))
        print("the r2 is: ", r2_score(y_test, y_predict))
        print("="*28)
        drawPlot(y_predict, y_test, 11, i)


# 预测多个设备号
originDataPath = "./"
devicesName = loadData(originDataPath)
buildModel(devicesName)
# 预测单个设备号
buildOneDevices(171170318052615)
#

print('\n'.join([''.join([('Love'[(x-y)%4]if((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3<=0 else' ')for x in range(-30,30)])for y in range(15,-15,-1)]))


