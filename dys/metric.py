from datetime import timedelta
import pandas as pd
from kupy import * 
from dys.alpha import Alphas
from dys.neutralize import Neutra
from dys.stockutil import existedBefore


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
        s = df.groupby('ticker')[args[1]].transform('pct_change', periods=args[0])
        # s = t[args[1]].pct_change(periods=args[0])
        return s

    def list_days(df: pd.DataFrame, args) ->pd.Series:
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
            raise Exception("list_days指标需要一个参数:1. 股票上市日期df，含ticker,list_date两个列")
        df_equ_list_days = args[0]
        df = pd.merge(df, df_equ_list_days, on='ticker', how='left')

        variant = df[df['list_date'].isna()]
        if variant.shape[0]>0:
            variant_export_file =configs["data_folder"].data + "/list_days_variant.csv"
            variant.to_csv(variant_export_file)
            logger.warning("发现找不到上市天数的股票{variant[['ticker','list_days']]} 有问题的数据保存到:{variant_export_file}")
            df.dropna(inplace=True)
        
        df["list_days"] = (df["trade_date"] - df["list_date"]).dt.days
        return df['list_days']

    def wq_alpha16(df: pd.DataFrame, args) ->pd.Series:
        stock = Alphas(df)
        def one_day_alpha(x):
            x['alpha16']=stock.alpha016()
            return x
        df = df.groupby('trade_date').apply(one_day_alpha)
        ds = df['alpha16']
        logger.debug(f"已调用Alpha016 数列")
        return ds


    def __one_day_ntra(x: pd.DataFrame, ntra:Neutra, col_name:str):
        """取一天行情为基础数据的中性化

        Args:
            x (pd.DataFrame): _description_
            ntra (Neutra): _description_

        Returns:
            _type_: _description_
        """        
        x["ntra_metric"] = ntra.apply(x[col_name])
        return x

    def ntra_turnover_rate(df: pd.DataFrame, args) ->pd.Series:
        ntra = Neutra(df["market_value"]) 
        df = df.groupby('trade_date').apply(m.__one_day_ntra, ntra, 'turnover_rate')
        ds = df['ntra_metric'] 
        return ds

    def ntra_bias(df: pd.DataFrame, args) ->pd.Series:
        ntra = Neutra(df["market_value"])
        ds = ntra.apply(df["bias"])
        df = df.groupby('trade_date').apply(m.__one_day_ntra, ntra, 'bias')
        ds = df['ntra_metric'] 
        return ds

    def bias(df: pd.DataFrame, args) ->pd.Series:
        """_summary_

        Args:
            df (pd.DataFrame): 输入df 是多股票日线集合，需要进行groupby操作 trade_date 由小到大排序
            args (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            pd.Series: bias的返回值为浮点数 比如：0.0084 等同于0.84%
        """        
        
        if len(args) != 1:
            raise Exception("bias(乖离率)指标需要一个参数:1. 均线天数")

        N = args[0]
        df = df.groupby('ticker').apply(m.__MA(N,'close_price'))
        ds = df['close_price']/df[f'MA{N}_close_price'] -1
        return ds

    def __SUM(x:pd.DataFrame, N:int, col_name:str)->pd.DataFrame:
        """滚动求和, 滚动顺序由x.index决定

        Args:
            x (pd.DataFrame): _description_
            N (int): 滚动N个元素

        Returns:
            pd.DataFrame: 返回求和列名为SUM{N}_{col_name}
        """        
        x[f'SUM{N}_{col_name}'] = x[col_name].rolling(N).sum()
        return x

    def __calc_float_num(df: pd.DataFrame, df_float: pd.DataFrame, N: int)->pd.DataFrame:
        fnN=f'SUM{N}_float_num'
        df_float = df_float[['ticker','float_date', 'float_num']]
        df_float["trade_date"]=df_float["float_date"]
        df = pd.merge(df, df_float, on=['ticker','trade_date'], how="left")
        # df.to_csv("/tmp/metric_float_rate_origin.csv")
        df = df.sort_values('trade_date', ascending=False).groupby('ticker').apply(m.__SUM, N, 'float_num')
        # df.to_csv("/tmp/metric_float_rate_descending.csv")
        # recover the sort order to origin
        df = df.sort_index()
        df[fnN].fillna(0, inplace=True)
        # df.to_csv("/tmp/metric_float_rate_ascending.csv")
        return df

    def float_value(df: pd.DataFrame, args) ->pd.Series:
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
            raise Exception("解禁比例指标需要两个参数:1. ,2 ")

        N = args[0]
        df_float=args[1]

        frN = f'float_rate_{N}'
        fvN = f'float_value_{N}'

        if frN in df.columns:
            df[fvN] = df[frN]*df['neg_market_value']
            return df[fvN]

        df = m.__calc_float_num(df,N,df_float)
        ds = df["float_num_" + str(N)]*10000*df["close_price"]
        return ds

    def float_rate(df: pd.DataFrame, args) ->pd.Series:
        if len(args) != 1:
            raise Exception("解禁金额指标需要两个参数:1. 天数,2 基本日线数字指标字段")
        
        frN = f'float_rate_{N}'
        fvN = f'float_value_{N}'

        N = args[0]
        df_float=args[1]
        if fvN in df.columns:
            df[frN] = df[fvN]/df['neg_market_value']
            return df[frN]

        df = m.__calc_float_num(df,N,df_float)
        ds = df["float_num_" + str(N)]*10000*df["close_price"]/df['neg_market_value']
        return ds
    
    def __MA(df:pd.DataFrame, N:int, col_name:str)->pd.DataFrame:
        """求N日平均值, 需要指定要求的均线指标名称col_name

        Args:
            df (pd.DataFrame): _description_
            N (int): _description_
            col_name: 要求的指标名称

        Returns:
            pd.DataFrame: 返回的新列名称MA{N}_{col_name}
        """        
        new_col_name=f'MA{N}_{col_name}'
        df[new_col_name]=df[col_name].rolling(N).mean()
        return df

        
    def vol_rate(df: pd.DataFrame, args) ->pd.DataFrame:
        """N日/M日 成交量能比率，都是正浮点数，用来看是否放量, 越大越放量，越小越缩量

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_
            1. N(int) 日
            2. M(int) 日

        Returns:
            pd.DataFrame: _description_
        """        
        N=args[0]
        M=args[1]
        df=df.groupby('ticker').apply(m.__MA(N,'vol'))
        df=df.groupby('ticker').apply(m.__MA(M,'vol'))
        ds = df[f'MA{N}_vol']/df[f'MA{M}_vol']
        return ds
        
    def chg_pct_sum(df: pd.DataFrame, args) ->pd.DataFrame:
        """N日内涨跌幅总和

        Args:
            df (pd.DataFrame): _description_
            args (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """        
        N=args[0]
        df=df.groupby('ticker').apply(m.__SUM(N,'chg_pct'))
        return df[f'SUM{N}_chg_pct']
