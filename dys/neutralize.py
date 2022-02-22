import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# 3倍中位数绝对偏差去极值
def mad(factor):
    me = np.median(factor)
    mad = np.median(abs(factor-me))
    # 求出3倍中位数的上下限制
    up = me + (3*1.4826*mad)
    down = me - (3*1.4826*mad)
    # 利用3倍中位数的值去极值
    factor = np.where(factor>up,up,factor)
    factor = np.where(factor<down,down,factor)
    return factor

# 实现标准化
def stand(factor):
    mean = factor.mean()
    std = factor.std()
    return (factor-mean)/std

class Neutra(object):
    def __init__(self, df):
        self.market_value = df["market_value"]
        self.pb = df["pb"]
        
