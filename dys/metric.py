import traceback
import warnings

import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from kupy import *

from dys.neutralize import Neutra


class m(object):
    """指标计算函数中第一个参数为固定参数，该dataframe 输入时候的排序方式是index
    输出Series的时候也一定要还原index的排序方式之后再输出,否则指标数据会错位

    Args:
        object (_type_): _description_
    """

    def ps_pit(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """增加每股收益，每股公积金，每股现金流指标，具体指标名称参考配置文档, 以及DBSchema字段解释
        @TODO
        有问题

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
            basic_eps 基本每股收益
            diluted_eps 稀释每股收益
            reser_ps 每股公积金
            c_reser_ps 每股资本公积金
            n_cin_cash_ps 每股现金流流量净额
            fcf_fps 每股企业活动现金流
        """
        # N = args[0]

        # def calc_open_price_prod(y):
        #     y["open_price_prod"] = (
        #         y["open_price"]
        #         .rolling(N, min_periods=1)
        #         .apply(lambda z: z.prod())
        #     )
        #     return y
        db = DBAdaptor(is_use_cache=True)
        df_ps = db.get_df_by_sql("select * from stock.fdmt_indi_ps_pit")
        df_ps = df_ps[
            [
                "ticker",
                "end_date",
                "act_pubtime",
                "basic_eps",
                "diluted_eps",
                "reser_ps",
                "c_reser_ps",
                "n_cin_cash_ps",
                "fcf_fps",
            ]
        ]

        df_ps.dropna(inplace=True)
        df_ps["trade_date"] = pd.to_datetime(df_ps["act_pubtime"]).dt.date
        df_ps.sort_values(
            ["ticker", "end_date", "trade_date"], ascending=True, inplace=True
        )
        df_ps.drop_duplicates(
            subset=["ticker", "end_date"], keep="first", inplace=True
        )
        df_ps.drop("end_date", axis=1, inplace=True)
        df_ps.drop("act_pubtime", axis=1, inplace=True)

        df_metric = pd.DataFrame(index=df.index)
        df_metric["ticker"] = df["ticker"]
        df_metric["trade_date"] = df["trade_date"]

        df_metric = pd.merge(
            df_metric, df_ps, on=["ticker", "trade_date"], how="left"
        )
        # df_metric[']

        df_metric.drop(["ticker", "trade_date"], axis=1, inplace=True)
        return df_metric

    def not_suspend_in(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日内是否有停牌 1为有，0为无

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        x = args[1]

        def calc_open_price_prod(y):
            y["open_price_prod"] = (
                y["open_price"]
                .rolling(N, min_periods=1)
                .apply(lambda z: z.prod())
            )
            return y

        x = x.groupby("ticker").apply(calc_open_price_prod)
        x[f"not_suspend_in_{N}"] = 1
        x.loc[x[f"open_price_prod"] == 0, f"not_suspend_in_{N}"] = 0
        df_metric = pd.DataFrame(index=df.index)
        df_metric["id"] = df["id"]
        df_metric = df_metric.join(x.iloc[:, -1])
        df_metric.drop("id", axis=1, inplace=True)
        return df_metric

    def roi_volat(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日收益波动率volatility

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]

        def volat(x):
            x["chg_pct_volat"] = x["chg_pct"].rolling(N).std()
            return x

        df_metric = pd.DataFrame(index=df.index)
        dx = df.groupby("ticker").apply(volat)
        df_metric[name] = dx["chg_pct_volat"]
        return df_metric

    def price_ampl(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """当日股价振幅

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = (df["highest_price"] - df["lowest_price"]) / df[
            "pre_close_price"
        ]
        return df_metric

    def price_ampl_rate(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """当日股价振幅比率, 股价振幅比率=（股价振幅)/当日成交额

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_metric = pd.DataFrame(index=df.index)
        # 因为这个比率太小，所以放大10000倍，最终要的是从大到小的排行
        df_metric[name] = df["price_ampl"] * 10000 / df["turnover_value"]
        return df_metric

    def ma_any(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """当日股价振幅

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        metric_name = args[1]

        if len(args) != 2:
            raise Exception("ma_any指标需要两个参数:1. 计算均值天数,2 要计算均值的指标名称")

        df_metric = pd.DataFrame(index=df.index)

        df = df.groupby("ticker").apply(m.__MA, N, metric_name, 1)
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = df.iloc[:, -1]

        return df_metric

    def sum_any(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """当日股价振幅

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        metric_name = args[1]

        if len(args) != 2:
            raise Exception("sum_any指标需要两个参数:1. 计算累计天数,2 要计算均值的指标名称")

        df_metric = pd.DataFrame(index=df.index)

        df = df.groupby("ticker").apply(m.__SUM, N, metric_name, 1)
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = df.iloc[:, -1]

        return df_metric

    def neg_market_amount(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """流通股数

        Args:
            df (pd.DataFrame): _description_
            name (_type_): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = (df["neg_market_value"] / df["close_price"]).astype(
            int
        )
        return df_metric

    def momentum(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日内变动百分比

        Args:
            df (pd.DataFrame): _description_
            args (:tuple): _description_
            1st: days --周期
            2nd: column_name


        Returns:
            pd.DataFrame: _description_
        """
        if df is None:
            raise Exception("数列不能为None")
        if len(args) != 2:
            raise Exception("momentum指标需要两个参数:1. 动量天数,2 基本日线数字指标字段")
        if args[0] < 1:
            raise Exception("计算动量因子需要大于1天")
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = df.groupby("ticker")[args[1]].transform(
            "pct_change", periods=args[0]
        )
        return df_metric

    def list_days(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """上市日期天数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 1:
            raise Exception(
                "list_days指标需要一个参数:1. 股票上市日期dataframe，含ticker,list_date两个列"
            )
        df_metric = pd.DataFrame(index=df.index)
        df_equ_list_info: pd.DataFrame = args[0]

        df_equ_list_info = df_equ_list_info[["ticker", "list_date"]]
        df_equ_list_info.set_index("ticker", inplace=True)
        # df = pd.merge(df, df_equ_list_info, on="ticker", how="left")

        # df['list_date'] = df['ticker'].apply(m.__add_list_date_col, args=(df_equ_list_info,))
        list_date_dict = df_equ_list_info.to_dict().get("list_date")
        df["list_date"] = df["ticker"].map(list_date_dict)

        variant = df[df["list_date"].isna()]
        if variant.shape[0] > 0:
            variant_export_file = (
                configs["data_folder"].data + "list_days_variant.csv"
            )
            logger.warning(
                f"发现找不到上市天数的股票{variant['ticker']} 有问题的数据保存到:{variant_export_file}"
            )
        
        

        df_metric[name] = (df["trade_date"] - df["list_date"]).dt.days + 1
        return df_metric

    def wq_alpha16(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """_wq alpha101 16号因子

        Args:
            df (pd.DataFrame): _description_
            args (_type_): 不需要参数

        Returns:
            pd.DataFrame: _description_
        """
        df_metric = pd.DataFrame(index=df.index)

        # 求每日中每个股票的价格与量各自的在市场的排名值
        df_t = df.groupby("trade_date").apply(m.__price_vol_rank_rate)
        # 求价，量 排行的协方差的方向值(即协方差值*-1），协方差为负值时，它越小(即负值绝对值越大，表示背离越严重)
        df_t = df_t.groupby("ticker").apply(m._negative_cov)
        df_t["wq_alpha16"] = df_t.groupby("trade_date")["neg_cov"].transform(
            "rank", method="max"
        )
        df_metric[name] = df_t["wq_alpha16"]
        # Debug inform
        logger.debug(f"Alpha016 sample数列已到处到/tmp/sample_wq_alpha16.csv")
        df_t.loc[
            (df_t["ticker"] == "603192")
            & (df_t["trade_date"] == pd.to_datetime("20210104")),
            :,
        ].to_csv("/tmp/sample_wq_alpha16.csv")

        return df_metric

    def ntra_turnover_rate(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """中性N日化换手率, 需要一个整数参数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. int N 日

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 1:
            raise Exception("list_days指标需要一个参数:1.N 日")

        df_metric = pd.DataFrame(index=df.index)

        N = args[0]
        col_name = f"MA{N}_turnover_rate"
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_rate")
        # 中性换手率不能有空值
        df[col_name] = df[col_name].fillna(0)
        df = df.groupby("trade_date").apply(m.__neutralize, col_name)
        df_metric[name] = df["ntra_metric"]
        return df_metric

    def ntra_bias(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """中性N日乖离率 需要一个整数参数

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. int N 日

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 1:
            raise Exception("list_days指标需要一个参数:1.N 日")

        df_metric = pd.DataFrame(index=df.index)
        N = args[0]
        col_name = f"MA{N}_bias"
        df_metric = m.bias(df, col_name, [N])
        df[col_name] = df_metric[col_name]
        # 中性换手率不能有空值
        df[col_name] = df[col_name].fillna(0)
        df = df.groupby("trade_date").apply(m.__neutralize, col_name)
        df_metric[name] = df["ntra_metric"]
        df_metric.drop(columns=[col_name], inplace=True)
        return df_metric

    def bias(df: pd.DataFrame, name, args) -> pd.DataFrame:
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

        df_metric = pd.DataFrame(index=df.index)

        N = args[0]
        df = df.groupby("ticker").apply(m.__MA, N, "close_price")
        df_metric[name] = df["close_price"] / df[f"MA{N}_close_price"] - 1
        return df_metric

    def float_value(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日内新增流通市值，单位万元

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1: N 日内, N日为N个交易日, 非自然日

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 2:
            raise Exception("解禁比例指标需要两个参数:1. 解禁前多少日受影响, 2 含停牌日df")

        df_metric = pd.DataFrame(index=df.index)

        N = args[0]
        x = args[1]

        frN = f"float_rate_{N}"
        fvN = f"float_value_{N}"
        fnN = "neg_shares_incr"

        if frN in df.columns:
            df[fvN] = df[frN] * df["neg_market_value"]
            return df

        x = m.__calc_float_num(x, N)
        df = df.join(x.iloc[:, -1])

        df_metric[name] = df[fnN] * df["close_price"] / 10000
        return df_metric

    def neg_share_incr(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日内新增流通股数,以万股为单位

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1: N 日内, N日为N个交易日, 非自然日

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 2:
            raise Exception("解禁比例指标需要两个参数:1. 解禁前多少日受影响， 2.含停牌日df")

        df_metric = pd.DataFrame(index=df.index)

        N = args[0]
        x = args[1]

        x = m.__calc_float_num(x, N)
        df = df.join(x.iloc[:, -1])

        df_metric[name] = df["neg_shares_incr"]
        return df_metric

    def float_rate(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日内解禁市值占流通市值的比例, 参数2个， 指标值为浮点小数, 单位 万分之一

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1: N 日内, N日为N个交易日, 非自然日

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 2:
            raise Exception("解禁金额指标需要两个参数:1. 天数,2,含停牌日的市场总行情数据")

        df_metric = pd.DataFrame(index=df.index)
        N = args[0]
        x = args[1]

        frN = f"float_rate_{N}"
        fvN = f"float_value_{N}"
        # fnN = f"SUM{N}_float_num"

        if fvN in df.columns:
            df[frN] = df[fvN] / df["neg_market_value"]
            return df

        x = m.__calc_float_num(x, N)
        df = df.join(x.iloc[:, -1])
        df_metric[name] = (
            df["neg_shares_incr"]
            * 10000
            * df["close_price"]
            / df["neg_market_value"]
        ).astype("int32") / 10000
        df_metric.loc[df_metric[name] < 0, name] = 0
        return df_metric

    def vol_nm_rate(df: pd.DataFrame, name, args) -> pd.DataFrame:
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

        df_metric = pd.DataFrame(index=df.index)
        N = args[0]
        M = args[1]
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_vol", 1)
        df = df.groupby("ticker").apply(m.__MA, M, "turnover_vol", 1)
        df_metric[name] = df[f"MA{N}_turnover_vol"] / df[f"MA{M}_turnover_vol"]
        df_metric.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_metric

    def vol_rate(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日均量同比

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N(int) 日

        Returns:
            pd.DataFrame: _description_
        """
        if len(args) != 1:
            raise Exception("N/M日量比指标需要一个参数:1. N天数")

        df_metric = pd.DataFrame(index=df.index)
        N = args[0]
        df = df.groupby("ticker").apply(m.__SUM, N, "turnover_vol", 1)
        df = df.groupby("ticker").apply(m.__SUM, 2 * N, "turnover_vol", 1)
        df_metric[name] = df[f"SUM{N}_turnover_vol"] / (
            df[f"SUM{2*N}_turnover_vol"] - df[f"SUM{N}_turnover_vol"]
        )
        df_metric.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_metric

    def revers_21(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """21日反转因子
        21日反转 :
        描述：
        公式：Product(收盘价/1030价格,21)
        详细说明： https://guorn.com/forum/post/p.783378.227880987472847
        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        db = DBAdaptor(is_use_cache=True)
        df_k1 = db.get_df_by_sql(
            "select ticker,trade_date,close from stock.mkt_equ_60k1 where trade_date >= '20200709' and trade_date<='20211231' and open>0 order by id"
        )
        df["merge_key"] = df[["ticker", "trade_date"]].apply(tuple, axis=1)
        df_k1["merge_key"] = df_k1[["ticker", "trade_date"]].apply(
            tuple, axis=1
        )
        df_k1 = df_k1[["merge_key", "close"]]
        df = pd.merge(df, df_k1, on="merge_key", how="left")

        df["rate"] = df["close_price"] / df["close"]
        df["rate"].fillna(1, inplace=True)

        def do_prod(df):
            df["revers_21"] = df["rate"].rolling(21).apply(lambda x: x.prod())
            return df

        df = df.groupby("ticker").apply(do_prod)
        df_metric = pd.DataFrame(index=df.index)

        df_metric[name] = df.iloc[:, -1]
        return df_metric

    def n_chg_pct(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日内涨跌幅总和
        注意这里和累积多日的涨跌幅是不同的；

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """

        N = args[0]
        x = args[1]
        x = x.groupby("ticker").apply(m.__caculate_offset_chg_pct, N, name)
        # df = df.groupby("ticker").apply(m.__caculate_offset_chg_pct, N )
        df_metric = pd.DataFrame(index=df.index)
        df_metric["id"] = df["id"]

        df_metric = df_metric.join(x.iloc[:, -1])
        df_metric.drop("id", axis=1, inplace=True)
        # df_metric[name] = df.iloc[:, -1]
        return df_metric

    def ma_turnover_rate(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日平均换手率 包含当前日
        deprecated 应该使用ma_any

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N日区间

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_rate", 1)

        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = df.iloc[:, -1]
        df_metric.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_metric

    def ma_vol(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日平均成交量 包含当前日
        deprecated 应该使用ma_any


        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N日区间

        Returns:
            pd.DataFrame: _description_
        """
        N = args[0]
        df = df.groupby("ticker").apply(m.__MA, N, "turnover_vol", 1)
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = df.iloc[:, -1]
        return df_metric

    def rank(df: pd.DataFrame, name, args) -> pd.DataFrame:
        """N日平均成交量 包含当前日

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N日区间

        Returns:
            pd.DataFrame: _description_
        """
        col_name = args[0]
        smallfirst = args[1]
        df_metric = pd.DataFrame(index=df.index)
        df_metric[name] = df.groupby("trade_date")[col_name].transform(
            "rank", ascending=smallfirst
        )
        return df_metric

    # Private Method
    def __price_vol_rank_rate(x: pd.DataFrame) -> pd.DataFrame:
        # Alphas为网上找的一个库，可能实现有问题，因此此处改为自己实现
        # 但可参考文档以及实现方式, 因此保留下面两行注释
        # stock = Alphas(x)
        # x["alpha16"] = stock.alpha016()
        x["highrank"] = x["highest_price"].rank()
        x["volrank"] = x["turnover_vol"].rank()
        return x

    def _negative_cov(x: pd.DataFrame, N: int = 5) -> pd.DataFrame:
        """取

        Args:
            x (_type_): _description_
        """
        # x['highrank'] = x['highrank'].rolling(N)
        # x['volrank'] = x['volrank'].rolling(N)
        x["neg_cov"] = -1 * x["highrank"].rolling(N).cov(x["volrank"])
        return x

    def __caculate_offset_chg_pct(x: pd.DataFrame, N: int, name):
        """计算N日内涨跌幅"""
        x[f"{N}_close_price"] = x["close_price"].shift(N)
        x[name] = (x["close_price"] - x[f"{N}_close_price"]) / x[
            f"{N}_close_price"
        ]
        x.drop(f"{N}_close_price", axis=1, inplace=True)
        return x

    def __neutralize(x: pd.DataFrame, col_name: str):
        """取一天行情为基础数据的中性化

        Args:
            x (pd.DataFrame): _description_
            ntra (Neutra): _description_

        Returns:
            _type_: _description_
        """

        ntra = Neutra(x["neg_market_value"])
        x["ntra_metric"] = np.nan
        try:
            x["ntra_metric"] = ntra.apply(x[col_name])
        except Exception as e:
            # spot_df = configs["data_folder"].data + "_neutralize_error.csv"
            # x.to_csv(spot_df)
            logger.error(f"中性化指标时候出错 Exception:{e}, col_name{col_name}")
            # Trace only
            # logger.error(traceback.format_exc())
            # logger.debug(f"中性化指标时候出错 Exception:{e}, col_name:{col_name} dataframe:{x}")
        return x

    def __SUM(
        x: pd.DataFrame, N: int, col_name: str, min_periods=-1
    ) -> pd.DataFrame:
        """滚动求和, 滚动顺序由x.index决定

        Args:
            x (pd.DataFrame): _description_
            N (int): 滚动N个元素

        Returns:
            pd.DataFrame: 返回求和列名为SUM{N}_{col_name}
        """
        if min_periods == -1 or min_periods >= N:
            min_periods = N - 1

        x[f"SUM{N}_{col_name}"] = (
            x[col_name].rolling(N, min_periods=min_periods).sum()
        )
        return x

    def __calc_float_num(df: pd.DataFrame, N: int) -> pd.DataFrame:
        # db = DBAdaptor(is_use_cache=True)
        # df_float = db.get_df_by_sql("select * from stock.equ_share_float")
        # fnN = f"SUM{N}_float_num"
        # df_float = df_float[["ticker", "float_date", "new_marketable_shares"]]
        # df_float["ticker"] = df_float["ticker"].astype("string")
        # df_float["float_date"] = pd.to_datetime(df_float["float_date"])
        # df_float.set_index(["ticker", "float_date"], inplace=True)
        # # df = pd.merge(df, df_float, on=["ticker", "trade_date"], how="left")
        # # df["ticker"] = df["ticker"].astype("string")
        # df["trade_date"] = pd.to_datetime(df["trade_date"])
        # df["float_num"] = df[["ticker", "trade_date"]].apply(tuple, axis=1)
        # float_num_dict = df_float.to_dict().get("new_marketable_shares")
        # df["float_num"] = df["float_num"].map(float_num_dict)
        # df["float_num"] = df["float_num"].fillna(0)
        # # df.to_csv("/tmp/metric_float_rate_origin.csv")
        df["neg_shares_1"] = (
            df["neg_market_value"] / df["close_price"]
        ).shift(1)
        df[f"neg_shares_-{N}"] = df["neg_shares_1"].shift(-1 * N)
        df["neg_shares_incr"] = df[f"neg_shares_-{N}"] - df[f"neg_shares_1"]

        # df = (
        #     df.sort_values("trade_date", ascending=False)
        #     .groupby("ticker")
        #     .apply(m.__SUM, N, "float_num", 1)
        # )
        # df.to_csv("/tmp/metric_float_rate_descending.csv")
        # recover the sort order to origin
        # df = df.sort_index()
        # TODO 这里要考虑90天的未来数据margin
        df["neg_shares_incr"] = df["neg_shares_incr"].fillna(0)
        # df.to_csv("/tmp/metric_float_rate_ascending.csv")
        return df

    def __MA(
        df: pd.DataFrame, N: int, col_name: str, min_periods=-1
    ) -> pd.DataFrame:
        """求N日平均值, 需要指定要求的均线指标名称col_name
        包含当日

        Args:
            df (pd.DataFrame): _description_
            N (int): _description_
            col_name: 要求的指标名称

        Returns:
            pd.DataFrame: 返回的新列名称MA{N}_{col_name}
        """
        if min_periods == -1 or min_periods >= N:
            min_periods = N - 1
        if N <= 1:
            raise Exception("N必须>1")

        new_col_name = f"MA{N}_{col_name}"
        df[new_col_name] = (
            df[col_name].rolling(N, min_periods=min_periods).mean()
        )
        return df
