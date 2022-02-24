import traceback
import pandas as pd
from kupy import *

from dys.alpha import Alphas
from dys.neutralize import Neutra


class m(object):
    """指标计算函数中第一个参数为固定参数，该dataframe 输入时候的排序方式是index
    输出Series的时候也一定要还原index的排序方式之后再输出,否则指标数据会错位

    Args:
        object (_type_): _description_
    """

    def momentum(df: pd.DataFrame, args) -> pd.Series:
        """N日内变动百分比

        Args:
            df (pd.DataFrame): _description_
            args (:tuple): _description_
            1st: days --周期
            2nd: column_name


        Returns:
            pd.Series: _description_
        """
        if df is None:
            raise Exception("数列不能为None")
        if len(args) != 2:
            raise Exception("momentum指标需要两个参数:1. 动量天数,2 基本日线数字指标字段")
        if args[0] < 1:
            raise Exception("计算动量因子需要大于1天")
        # t = df.copy()
        s = df.groupby("ticker")[args[1]].transform(
            "pct_change", periods=args[0]
        )
        # s = t[args[1]].pct_change(periods=args[0])
        return s

    def list_days(df: pd.DataFrame, args) -> pd.Series:
        """上市日期天数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            pd.Series: _description_
        """
        if len(args) != 1:
            raise Exception(
                "list_days指标需要一个参数:1. 股票上市日期dataframe，含ticker,list_date两个列"
            )
        df_equ_list_info :pd.DataFrame = args[0]

        df_equ_list_info = df_equ_list_info[
            ["ticker", "list_date"]
        ]
        df_equ_list_info.set_index('ticker', inplace=True)
        # df = pd.merge(df, df_equ_list_info, on="ticker", how="left")

        # df['list_date'] = df['ticker'].apply(m.__add_list_date_col, args=(df_equ_list_info,))
        list_date_dict =df_equ_list_info.to_dict().get('list_date')
        df['list_date'] = df['ticker'].map(list_date_dict)

        variant = df[df["list_date"].isna()]
        if variant.shape[0] > 0:
            variant_export_file = (
                configs["data_folder"].data + "/list_days_variant.csv"
            )
            logger.warning(
                f"发现找不到上市天数的股票{variant['ticker']} 有问题的数据保存到:{variant_export_file}"
            )

        df["list_days"] = (df["trade_date"] - df["list_date"]).dt.days + 1
        return df["list_days"]

    def __alpha(x):
        stock = Alphas(x)
        x["alpha16"] = stock.alpha016()
        return x

    def wq_alpha16(df: pd.DataFrame, args) -> pd.Series:
        """_wq alpha101 16号因子

        Args:
            df (pd.DataFrame): _description_
            args (_type_): 不需要参数

        Returns:
            pd.Series: _description_
        """

        df = df.groupby("ticker").apply(m.__alpha)
        ds = df["alpha16"]
        logger.debug(f"已调用Alpha016 数列")
        return ds

    def __neutralize(x: pd.DataFrame,  col_name: str):
        """取一天行情为基础数据的中性化

        Args:
            x (pd.DataFrame): _description_
            ntra (Neutra): _description_

        Returns:
            _type_: _description_
        """

        ntra = Neutra(x["neg_market_value"])
        try:
            x["ntra_metric"] = ntra.apply(x[col_name])
        except Exception as e:
            spot_df = configs["data_folder"].data + "_neutralize_error.csv"
            # x.to_csv(spot_df)
            logger.error(f"中性化指标时候出错 Exception:{e}, col_name{col_name}")
            #Trace only
            # logger.error(traceback.format_exc())
            # logger.debug(f"中性化指标时候出错 Exception:{e}, col_name:{col_name} dataframe:{x}")
        return x

    def ntra_turnover_rate(df: pd.DataFrame, args) -> pd.Series:
        """中性N日化换手率, 需要一个整数参数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. int N 日

        Returns:
            pd.Series: _description_
        """
        if len(args) != 1:
            raise Exception(
                "list_days指标需要一个参数:1.N 日" 
            )
        N = args[0]
        col_name=f'MA{N}_turnover_rate'
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_rate")
        df[col_name].fillna(df.iat[N-1,df.shape[1]-1], inplace=True)
        # df[col_name].fillna(0, inplace=True)
        # logger.debug(df)
        df = df.groupby("trade_date").apply(m.__neutralize, col_name)
        ds = df["ntra_metric"]
        return ds

    def ntra_bias(df: pd.DataFrame, args) -> pd.Series:
        """中性N日乖离率 需要一个整数参数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. int N 日

        Returns:
            pd.Series: _description_
        """
        if len(args) != 1:
            raise Exception(
                "list_days指标需要一个参数:1.N 日" 
            )
        N = args[0]
        col_name=f'MA{N}_bias'
        df[col_name] = m.bias(df, [N])
        df[col_name].fillna(df.iat[N-1,df.shape[1]-1], inplace=True)
        df = df.groupby("trade_date").apply(m.__neutralize, col_name)
        ds = df["ntra_metric"]
        return ds

    def bias(df: pd.DataFrame, args) -> pd.Series:
        """乖离率 收盘价偏离N日均线差值，负数越小表示越有反弹可能
        一个参数 N日

        Args:
            df (pd.DataFrame): 输入df 是多股票日线集合，需要进行groupby操作 trade_date 由小到大排序
            args (_type_): _description_
            1. int N 日

        Raises:
            Exception: _description_

        Returns:
            pd.Series: bias的返回值为浮点数 比如：0.0084 等同于0.84%
        """

        if len(args) != 1:
            raise Exception("bias(乖离率)指标需要一个参数:1. 均线天数")

        N = args[0]
        df = df.groupby("ticker").apply(m.__MA, N, "close_price")
        ds = df["close_price"] / df[f"MA{N}_close_price"] - 1
        return ds

    def __SUM(x: pd.DataFrame, N: int, col_name: str) -> pd.DataFrame:
        """滚动求和, 滚动顺序由x.index决定

        Args:
            x (pd.DataFrame): _description_
            N (int): 滚动N个元素

        Returns:
            pd.DataFrame: 返回求和列名为SUM{N}_{col_name}
        """
        x[f"SUM{N}_{col_name}"] = x[col_name].rolling(N, min_periods=1).sum()
        return x

    def __(x, y: pd.DataFrame):
        d = y.loc[y.ticker == x,'list_date'].iloc[0]
        logger.debug(f"{x} {d} ")
        return d

    def __calc_float_num(
        df: pd.DataFrame, N: int, df_float: pd.DataFrame
    ) -> pd.DataFrame:
        fnN = f"SUM{N}_float_num"
        df_float = df_float[["ticker", "float_date", "float_num"]]
        # df_float['ticker'] = df_float['ticker'].astype('string')
        # df_float['float_date'] = pd.to_datetime(df_float['float_date'])
        df_float.set_index(['ticker', 'float_date'], inplace=True)
        # df = pd.merge(df, df_float, on=["ticker", "trade_date"], how="left")
        df['ticker']=df['ticker'].astype('string')
        # df.groupby(['ticker','trade_date'])[''].map
        # df['trade_date'] = pd.to_datetime(df['trade_date'])
        df["float_num"]=df[['ticker', 'trade_date']].apply(tuple, axis=1)
        float_num_dict = df_float.to_dict().get('float_num')
        df['float_num'] = df['float_num'].map(float_num_dict)
        df['float_num'].fillna(0,inplace=True)
        # df.to_csv("/tmp/metric_float_rate_origin.csv")
        df = (
            df.sort_values("trade_date", ascending=False)
            .groupby("ticker")
            .apply(m.__SUM, N, "float_num")
        )
        # df.to_csv("/tmp/metric_float_rate_descending.csv")
        # recover the sort order to origin
        df = df.sort_index()
        df[fnN].fillna(0, inplace=True)
        # df.to_csv("/tmp/metric_float_rate_ascending.csv")
        return df

    def float_value(df: pd.DataFrame, args) -> pd.Series:
        """N日内解禁市值，单位元

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1: N 日内, N日为N个交易日, 非自然日
            2: 解禁df

        Raises:
            Exception: _description_

        Returns:
            pd.Series: _description_
        """
        if len(args) != 2:
            raise Exception("解禁比例指标需要两个参数:1. 解禁前多少日受影响 ,2 含解禁信息的dataframe ")

        N = args[0]
        df_float = args[1]

        frN = f"float_rate_{N}"
        fvN = f"float_value_{N}"
        fnN = f"SUM{N}_float_num"

        if frN in df.columns:
            df[fvN] = df[frN] * df["neg_market_value"]
            return df[fvN]

        df = m.__calc_float_num(df,  N, df_float)
        ds = df[fnN] * 10000 * df["close_price"]
        return ds

    def float_rate(df: pd.DataFrame, args) -> pd.Series:
        """N日内解禁市值占流通市值的比例, 参数2个， 指标值为浮点小数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1: N 日内, N日为N个交易日, 非自然日
            2: 解禁df

        Raises:
            Exception: _description_

        Returns:
            pd.Series: _description_
        """
        if len(args) != 2:
            raise Exception("解禁金额指标需要两个参数:1. 天数,2 基本日线数字指标字段")

        N = args[0]
        df_float = args[1]

        frN = f"float_rate_{N}"
        fvN = f"float_value_{N}"
        fnN = f"SUM{N}_float_num"

        if fvN in df.columns:
            df[frN] = df[fvN] / df["neg_market_value"]
            return df[frN]

        df = m.__calc_float_num(df, N, df_float)
        ds = df[fnN] * 10000 * df["close_price"] / df["neg_market_value"]
        return ds

    def __MA(df: pd.DataFrame, N: int, col_name: str) -> pd.DataFrame:
        """求N日平均值, 需要指定要求的均线指标名称col_name
        包含当日

        Args:
            df (pd.DataFrame): _description_
            N (int): _description_
            col_name: 要求的指标名称

        Returns:
            pd.DataFrame: 返回的新列名称MA{N}_{col_name}
        """
        if N <= 1:
            raise Exception("")
        new_col_name = f"MA{N}_{col_name}"
        df[new_col_name] = df[col_name].rolling(N).mean()
        return df

    def vol_rate(df: pd.DataFrame, args) -> pd.DataFrame:
        """N日/M日 成交量能比率，都是正浮点数，用来看是否放量, 越大越放量，越小越缩量

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N(int) 日
            2. M(int) 日

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 2:
            raise Exception("N/M日量比指标需要两个参数:1. N天数,2 M天数")
        N = args[0]
        M = args[1]
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_vol")
        df = df.groupby("ticker").apply(m.__MA, M, "turnover_vol")
        ds = df[f"MA{N}_turnover_vol"] / df[f"MA{M}_turnover_vol"]
        return ds

    def sum_chg_pct(df: pd.DataFrame, args) -> pd.DataFrame:
        """N日内涨跌幅总和

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        df = df.groupby("ticker").apply(m.__SUM, N, "chg_pct")
        return df[f"SUM{N}_chg_pct"]

    def ma_turnover_rate(df: pd.DataFrame, args) -> pd.DataFrame:
        """N日平均换手率 包含当前日

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N日区间

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_rate")
        return df[f"MA{N}_turnover_rate"]

    def ma_vol(df: pd.DataFrame, args) -> pd.DataFrame:
        """N日平均换手率 包含当前日

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N日区间

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_vol")
        return df[f"MA{N}_turnover_vol"]



