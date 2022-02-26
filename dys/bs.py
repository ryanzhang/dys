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
import os
from typing import Any, Iterable, List

import pandas as pd
from kupy.config import configs
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.domain import RankFactor, SelectMetric, StrategyConfig, TradeModel
from dys.stockutil import StockUtil

# By default 120 tradeable days between 2009-07-09 to 2020-12-31
# New k data can be append since 2022-01-04
DEFAULT_TOTAL_START_DATE= '20090709'
DEFAULT_TOTAL_END_DATE='20211231'
class BaseStrategy:
    def __init__(self,  start_date: str = DEFAULT_TOTAL_START_DATE, end_date: str= DEFAULT_TOTAL_END_DATE, trade_type: int = 0 ):
        logger.info("Construct dys Base")
        self.dataset_start = start_date
        self.dataset_end = end_date
        self.trade_type = trade_type  # 0表示股票， 1表示基金

        # 成员
        self.df_equ_pool = None
        self.df_equd_pool = None

        self.select_conditions = None
        self.debug_sample_date = None
        self.trade_model = None
        self.rank_factors = None
        # self.select_metrics = None
        self.df_metrics = None

        self.mkt_timing_algo = None
        # 历史持仓
        self.df_position = None
        # 过滤出来的股票池
        self.df_choice_equd = None

        # Strategy Config
        self.config = StrategyConfig()
        self.__mk_folder()
        # 根据添加的指标滚动窗口，记录最大向前滚动的时间窗口 以保证边缘日期的均值计算问题
        self.start_date = None
        self.end_date = None

        self.__load_total_data_set()

        self.df_choice_equd = self.df_equd_pool

    def __set_cache_folder(self, path):
        self.config.data_folder = path
        self.__mk_folder()

    def __mk_folder(self ):
        os.makedirs(self.config.data_folder+"cache/", exist_ok=True)
        os.makedirs(self.config.data_folder+"metrics/", exist_ok=True)

    def __load_total_data_set(self):
        db = DBAdaptor(is_use_cache = True)
        if self.trade_type == 0:
            self.df_equ_pool = db.get_df_by_sql(
                "select * from stock.equity"
            )
            logger.debug(f"股票池已经加载所有上市股票")
        elif self.trade_type == 1:
            self.df_equ_pool = db.get_df_by_sql(
                "select * from stock.fund where etf_lof='ETF'"
            )
            logger.debug(f"股票池已经加载所有ETF基金")
        if self.df_equ_pool.shape[0] == 0:
            raise Exception("从数据库加载股票|基金数据为空!")
        logger.debug(f"股票池已经加载所有股票数据:{self.df_equ_pool.shape[0]}")

        if self.trade_type == 0:
            self.df_equd_pool = db.get_df_by_sql(
                f"select * from stock.mkt_equ_day where trade_date >='{self.dataset_start}' and trade_date <= '{self.dataset_end}' and open_price > 0"
            )
            logger.debug(f"股票池已经设定所有上市股票")
        elif self.trade_type == 1:
            self.df_equd_pool = db.get_df_by_sql(
                f"select * from stock.fund_day  where trade_date >='{self.dataset_start}'  and trade_date <= '{self.dataset_end}' and open_price > 0"
            )
            logger.debug(f"股票池已经设定所有ETF基金")
        if self.df_equd_pool.shape[0] == 0:
            raise Exception("从数据库加载日线数据为空!")
        logger.debug(f"股票池已经加载所有日线数据:{self.df_equd_pool.shape[0]}")
    

    def select_equd_by_equ_pool(self):
        df = self.df_equd_pool
        oc = df.shape[0]
        df = df[df["ticker"].isin(self.df_equ_pool["ticker"])]
        self.df_choice_equd = df
        nc = self.df_choice_equd.shape[0]
        logger.debug(f"K线数据已根据股票池更新{self.df_choice_equd.shape[0]}, 过滤掉{nc-oc}")
        return df

    def append_select_condition(self, condition):
        if self.select_conditions is None:
            self.select_conditions = list()

        if self.debug_sample_date is not None:
            oc = self.df_choice_equd[
                self.df_choice_equd.trade_date == self.debug_sample_date
            ].shape[0]

        self.df_choice_equd = self.df_choice_equd.query(condition)

        if self.debug_sample_date is not None:
            nc = self.df_choice_equd[
                self.df_choice_equd.trade_date == self.debug_sample_date
            ].shape[0]

        self.select_conditions.append(condition)
        if self.debug_sample_date is not None:
            logger.debug(
                f"已加载条件{condition}, {self.debug_sample_date} 现存:{nc} 过滤掉: {nc-oc}个股票"
            )

    def append_metric(self, sm: SelectMetric, reset_cache:bool=False):
        self.df_choice_equd[sm.name] = self.__add_metric_column(sm, reset_cache)

        # Debug information
        if self.debug_sample_date is not None:
            df = self.df_choice_equd.loc[
                self.df_choice_equd.trade_date == self.debug_sample_date,
                ["ticker", sm.name],
            ]
            logger.debug(f"指标{sm.name}已加载, {self.debug_sample_date} 指标值: {df}")

    def append_metrics(self, sms: List[SelectMetric]):
        for sm in sms:
            self.append_metric(sm)

    def __add_metric_column(self, sm: SelectMetric, reset_cache:bool =False) -> pd.Series:
        metric_cache_file = f'{self.config.data_folder}/metrics/{sm.name}.paquet' 

        if reset_cache or not os.path.exists(metric_cache_file):
            # Compute the metric 
            df_metric = sm.apply(self.df_choice_equd, sm.name, sm.args)
            df_metric.to_parquet(metric_cache_file, index=True)
        else:
            # Load from file
            df_metric = pd.read_parquet(metric_cache_file)

        logger.debug(f"指标{sm.name}已经加载到df_choice_equd 列表中")
        return df_metric[sm.name]

    def post_hook_select_equ(self):
        pass

    def append_rankfactor(self, rf: RankFactor):
        cols = self.df_choice_equd.columns

        if rf.name not in cols:
            raise Exception(f"排序指标没有在df_choice_equd列中找到:{cols}")

        if self.rank_factors is None:
            self.rank_factors = list()
        self.rank_factors.append(rf)

    def append_rankfactors(self, rfs: List[RankFactor]):
        if self.rank_factors is None:
            self.rank_factors = list()

        self.rank_factors.extend(rfs)

    def rank(self) -> pd.DataFrame:
        # 再次根据时间过滤一次日线数据 ，因为之前可能考虑了计算指标时候的margin 数据要去除

        df: pd.DataFrame = self.df_choice_equd
        df["rank"] = 0
        tw = 0
        for rf in self.rank_factors:
            if rf.name not in df.columns:
                raise Exception(
                    f"排序因子{rf.name}无法在df_choice_equd的列中找到:{df.columns}"
                )
            subrank_column = rf.name + "_subrank"
            df[subrank_column] = (
                df.groupby("trade_date")[rf.name].transform(
                    "rank", ascending=rf.bigfirst, pct=True, method="max"
                )
                * 100
            )
            df["rank"] = df["rank"] + df[subrank_column] * rf.weight
            tw = tw + rf.weight

        df["rank"] = df["rank"] / tw

        # 转换为百分制
        df["xrank"] = (
            df.groupby("trade_date")["rank"].transform(
                "rank", ascending=True, pct=True, method="max"
            )
            * 100
        )

        df.sort_values(["trade_date", "xrank"], ascending=False, inplace=True)

        self.df_choice_equd = df

        if self.debug_sample_date is not None:
            df = self.df_choice_equd.loc[
                self.df_choice_equd.trade_date == self.debug_sample_date, :
            ]
            logger.debug(f"排序已完成, {self.debug_sample_date}  指标值: {df}")
        logger.debug(f"选好的股票已经排序")
        return df


    def select_equd_by_date(
        self, start_date: date, end_date: date = None
    ) -> pd.DataFrame:

        df = self.df_choice_equd

        if end_date is None:
            df = df[(df.trade_date >= start_date)]
        else:
            df = df[
                (df.trade_date >= start_date) & (df.trade_date <= end_date)
            ]
        self.df_choice_equd = df

        return df

    def generate_trade_mfst(
        self, start_date: date, end_date: date = None
    ) -> pd.DataFrame:
        mfst = pd.DataFrame()
        # df = self.df_choice_equd

        return mfst

    def get_choice_equ_list_by_date(self, trade_date: date) -> pd.DataFrame:
        df = self.df_choice_equd
        df.loc["trade_date" == trade_date, ""]
        logger.debug(f"{trade_date} 选股已经返回")
        pass

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

        self.calc_choice_equd(start_date, end_date)

        mfst = self.generate_trade_mfst(start_date, end_date)

        logger.debug(f"回测结果清单已经生成{mfst}")

        return mfst

    def get_roi_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("策略收益统计信息已返回")
        return pd.DataFrame()

    def get_trade_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("交易统计已返回")

        return pd.DataFrame()

    def get_rebalance_operation(
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
