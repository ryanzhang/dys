import pandas as pd
from kupy import * 
from dys.alpha import Alphas


class m(object):
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
        s = df[args[1]].pct_change(periods=args[0])
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
            raise Exception("list_days指标需要两个参数:1. 动量天数,2 基本日线数字指标字段")
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
        ds = stock.alpha016()
        logger.debug(f"已调用Alpha016 数列")
        return ds

    def ntra_turnover_rate(df: pd.DataFrame, args) ->pd.Series:
        
        return ds
        