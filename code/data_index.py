import pandas as pd
import numpy as np


def Annualreturn(return_list):
    ret = np.array(return_list)
    ret = (ret[-1] - ret[0]) / ret[0]
    annualret = ret / len(return_list) * 252
    return annualret


def Maxdrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    j = np.argmax(return_list[:i])  # 开始位置
    return (return_list[j] - return_list[i]) / (return_list[j])


def Sharperatio(rtn, period):
    rtn = rtn.diff() / rtn.shift(1)[1:]
    sp = rtn.mean() * (period ** 0.5) / rtn.std()
    return sp


def getNewPrice(prices,ratio):
    newP = np.sum(prices*ratio,axis=1)
    return newP


def getRatio(profile,name,delt):
    prices = {}
    for i in range(len(profile)):
        df = profile[i]
        prices[i] = df[price00]
    prices = pd.DataFrame(prices)
    prices.columns = name

    prices1 = {}
    for j in range(len(profile)):
        df1 = profile[j]
        prices1[j] = df1[price01]
    prices1 = pd.DataFrame(prices1)
    prices1.columns = name
    
