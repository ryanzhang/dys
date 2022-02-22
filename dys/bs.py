# -*- coding: UTF-8 -*-
"""
dys base module.

This is the principal module of the kupy project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""


from datetime import date
from typing import Any, Callable, List

import pandas as pd
from kupy.config import configs
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.domain import RankFactor, SelectMetric, TradeModel


class BaseStrategy:
    def __init__(self, trade_type: int = 0):
        logger.info("Construct dys Base")
        self.trade_type = trade_type  # 0表示股票， 1表示基金
        is_use_df_cache = bool(configs["is_use_df_cache"].data)
        self.__db = DBAdaptor(is_use_cache=is_use_df_cache)

        self.df_equ_pool: Any = None
        self.df_equd_pool: Any = None

        self.select_equ_condition: Any = None
        self.trade_model: Any = None
        self.rank_factors: Any = None
        self.select_metrics: Any = None
        self.mkt_timing_algo: Any = None
        # 历史持仓
        self.df_position: Any = None
        # 每日择股
        self.df_choice_equd: Any = None

    def load_all_equ(self) -> pd.DataFrame:
        """默认加载所有股票或者etf基金, 取决于trade_type

        Returns:
            pd.DataFrame: _description_
        """
        if self.trade_type == 0:
            self.df_equ_pool = self.__db.get_df_by_sql("select * from stock.equity ")
            logger.debug(f"股票池已经设定所有上市股票")
        elif self.trade_type == 1:
            self.df_equ_pool = self.__db.get_df_by_sql(
                "select * from stock.fund where etf_lof='ETF' "
            )
            logger.debug(f"股票池已经设定所有ETF基金")
        if self.df_equ_pool.shape[0] == 0:
            raise Exception("从数据库加载股票|基金数据为空!")
        logger.debug(f"股票池已经加载所有股票数据:{self.df_equ_pool.shape[0]}")
        return True

    def load_all_equd(self) -> pd.DataFrame:
        """默认加载所有股票或者etf基金, 取决于trade_type

        Returns:
            pd.DataFrame: _description_
        """
        if self.trade_type == 0:
            self.df_equd_pool = self.__db.get_df_by_sql("select * from stock.mkt_equ_day where trade_date >'20100101'")
            logger.debug(f"股票池已经设定所有上市股票")
        elif self.trade_type == 1:
            self.df_equd_pool = self.__db.get_df_by_sql(
                "select * from stock.fund_day where trade_date > '20100101'"
            )
            logger.debug(f"股票池已经设定所有ETF基金")
        if self.df_equd_pool.shape[0] == 0:
            raise Exception("从数据库加载日线数据为空!")
        logger.debug(f"股票池已经加载所有日线数据:{self.df_equd_pool.shape[0]}")
        return True

    def append_metric(self, sm:SelectMetric):
        if self.select_metrics is None:
            self.select_metrics = list()

        self.select_metrics.append(sm)
        
    def append_metrics(self, sms:SelectMetric):
        if self.select_metrics is None:
            self.select_metrics = list()

        self.select_metrics.extend(sms)
        
    def __add_metric_column(
        self
    ) -> pd.DataFrame:
        if self.select_metrics is None:
            pass

        df = self.df_choice_equd
        for sm in self.select_metrics:
            df[sm.name] = sm.apply(df,sm.args)

        self.df_choice_equd = df
        logger.debug(f"选股指标已经加载到df_choice_equd 列表中 {self.df_choice_equd}")
        return df
    
    def post_hook_select_equ(self):
        pass

    def append_rankfactor(self, rf:RankFactor):
        cols=self.df_choice_equd.columns
        if rf.name not in cols:
            raise Exception(f"排序指标没有在df_choice_equd列中找到:{cols}")

        if self.rank_factors is None:
            self.rank_factors = list()

        self.rank_factors.append(rf)
        
    def append_rankfactors(self, rfs:List[RankFactor]):
        if self.rank_factors is None:
            self.rank_factors = list()

        self.rank_factors.extend(rfs)
        
    def rank(self) -> pd.DataFrame:
        df:pd.DataFrame = self.df_choice_equd
        df['rank'] =0
        tw=0
        for rf in self.rank_factors:
            if rf.name not in df.columns:
                raise Exception(f"排序因子无法在df_choice_equd的列中找到:{df.columns}")
            subrank_column = rf.name + "_subrank"
            df[subrank_column]=df.groupby('trade_date')[rf.name].transform('rank', ascending=rf.bigfirst, pct=True)*100
            # 转换为百分制
            df['rank']=df['rank'] +df[subrank_column]*rf.weight
            df.sort_values(['trade_date', 'rank'], ascending=False, inplace=True)
            tw=tw + rf.weight
        df['rank'] = df['rank']/tw
        self.df_choice_equd = df

        logger.debug(f"选好的股票已经排序")
        return df

    def __select_equd_by_date(
        self,
        start_date: date,
        end_date: date = None,
    ) -> pd.DataFrame:
        logger.debug(f"Select between {start_date} - {end_date}")

        if self.df_equ_pool is None:
           self.load_all_equ() 
        if self.df_equd_pool is None:
            self.load_all_equd()

        df:pd.DataFrame = self.df_equd_pool
        if end_date is None:
            df = df[(df.trade_date >= start_date)]
        else:
            df = df[(df.trade_date >= start_date)&(df.trade_date<=end_date)]
        df = df[df.ticker.isin(self.df_equ_pool.ticker)]
        # df.reset_index(inplace=True)
        self.df_choice_equd = df
        return df

    def __select_equd_by_expression (
        self
    ) -> pd.DataFrame:
        logger.debug(f"正在根据指标条件选股:{self.select_equ_condition} ")

        df:pd.DataFrame = self.df_choice_equd
        if df is None:
            raise Exception(f"df_choice_equd还没有设置")

        select_condition = self.select_equ_condition
        if select_condition is not None and select_condition != "":
            df = df.query(self.select_equ_condition)
        self.df_choice_equd = df
        return df


    def generate_trade_mfst(self, start_date:date, end_date:date):    
        mfst = pd.DataFrame()
        df = self.df_choice_equd
        
        

        return tm 

    def get_roi_mfst(
        self, start_date: date, end_date: date = None
    ) -> pd.DataFrame:
        """根据时间段给出回测数据清单, 大致回测流程
        #1. 设置交易模型
        #2. 设置交易模型
        #3. 大盘择时
        #4. 回测

        Args:
            start_date (date): _description_
            end_date (date): _description_

        Returns:
            pd.DataFrame: _description_
        """

        # 根据日期筛选
        self.__select_equd_by_date( start_date, end_date)

        # 扩充自定义指标列
        self.__add_metric_column()

        df = self.__select_equd_by_expression()

        logger.debug(f"根据择股条件选股已经生效:{df.ticker.nunique()}")


        df = self.rank()
        logger.debug(f"选择的股票，每日行情排序已经完成:{df[['ticker','rank']]} ")

        mfst = self.generate_trade_mfst(start_date, end_date)

        logger.debug(f"回测结果清单已经生成{mfst}")

        return mfst

    def get_roi_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("策略收益统计信息已返回")
        return pd.DataFrame()

    def get_trade_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("交易统计已返回")

        return pd.DataFrame()

    def get_choice_equ(self, trade_date: date) -> pd.DataFrame:
        """根据给定的日期，获取根据策略选股条件以及排序之后选出的股票

        Args:
            trade_date (date): _description_

        Returns:
            list: _description_
        """
        logger.debug(f"{trade_date} 选股已经返回")
        return pd.DataFrame()

    def get_exchange_trans(
        self, trade_date: date, start_date: date = None, end_date: date = None
    ) -> pd.DataFrame:
        """返回调仓指令

        Returns:
            dataframe: 返回某日的调仓指令
        """

        return pd.DataFrame()

    def get_sale_trans_by_date(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """_summary_

        Args:
            start_date (date): _description_
            end_date (date): _description_

        Returns:
            pd.DataFrame: _description_
        """
        logger.debug(f"{start_date}-{end_date} 卖出交易已返回")
        return pd.DataFrame()

    def get_buy_trans_by_date(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """_summary_

        Args:
            start_date (date): _description_
            end_date (date): _description_

        Returns:
            pd.DataFrame: _description_
        """
        logger.debug(f"{start_date}-{end_date} 买入交易已返回")
        return pd.DataFrame()

    def show_plot(self, df: pd.DataFrame):
        """根据start_date, end_date 返回

        Args:
            start_date (date): _description_
            end_date (date): _description_

        Returns:
            str: _description_
        """
        logger.debug("收益曲线已经显示")
        pass
        return pd.DataFrame()
