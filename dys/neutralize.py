import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# 3倍中位数绝对偏差去极值
def mad(factor: pd.Series) -> pd.Series:
    me = np.median(factor)
    mad = np.median(abs(factor - me))
    # 求出3倍中位数的上下限制
    up = me + (3 * 1.4826 * mad)
    down = me - (3 * 1.4826 * mad)
    # 利用3倍中位数的值去极值
    factor = np.where(factor > up, up, factor)
    factor = np.where(factor < down, down, factor)
    return factor


# 实现标准化
def stand(factor: pd.Series) -> pd.Series:
    mean = factor.mean()
    std = factor.std()
    return (factor - mean) / std


class Neutra(object):
    def __init__(self, ds_market_value):
        self.market_value = ds_market_value

    def __standalize(self, target: pd.Series) -> pd.Series:
        target = mad(target)
        target = stand(target)
        return target

    def __linear_regression(self, target: pd.Series) -> pd.Series:
        x = self.market_value.to_frame()
        y = target
        lr = LinearRegression()
        lr.fit(x, y)
        y_predict = lr.predict(x)
        return y_predict

    def apply(self, target: pd.Series) -> pd.Series:
        """返回残差

        Args:
            target (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        target = self.__standalize(target)
        predict = self.__linear_regression(target)
        return target - predict

    def standalize(self, target: pd.Series) -> pd.Series:
        return self.__standalize(target)

    def predict(self, target: pd.Series) -> pd.Series:
        target = self.__standalize(target)
        return self.__linear_regression(target)
