import pandas as pd


class m(object):
    def momentum( df: pd.DataFrame, args) -> pd.Series:
        if df is None:
            raise Exception("数列不能为None")
        if len(args) != 2:
            raise Exception("mementum指标需要两个参数:1. 动量天数,2 基本日线数字指标字段")
        if args[0] < 1:
            raise Exception("计算动量因子需要大于1天")
        s = df[args[1]].pct_change(periods=args[0])
        return s
