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

import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from datetime import date, datetime, timedelta
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from kupy.config import configs
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.domain import RankFactor, SelectMetric, StrategyConfig, TradeModel
from dys.stockutil import StockUtil

# By default 120 tradeable days between 2009-07-09 to 2020-12-31
# New k data can be append since 2022-01-04
# 目前回测数据只支持到2021/12/31日

DEFAULT_TOTAL_START_DATE = "20090709"
DEFAULT_TOTAL_END_DATE = "20211231"


class BaseStrategy:
    def __init__(
        self,
        start_date: str = DEFAULT_TOTAL_START_DATE,
        end_date: str = DEFAULT_TOTAL_END_DATE,
        trade_type: int = 0,
    ):
        logger.info("Construct dys Base")
        self.dataset_start = start_date
        self.dataset_end = end_date
        self.trade_type = trade_type  # 0表示股票， 1表示基金

        # 成员
        self.df_equ_pool = None
        self.df_equd_pool = None
        self.df_equd_pool_with_sus = None
        self.equd_groupby_date = None

        self.select_conditions = None
        self.debug_sample_date = None
        self.trade_model = None
        self.rank_factors = None
        # self.select_metrics = None
        self.df_metrics: List(str) = list()

        self.mkt_timing_algo = None
        # 历史持仓
        # columns_pos_mfst = [
        #     "period",
        #     "sec_short_name",
        #     "ticker",
        #     "start_date",
        #     "end_date",
        #     "period_pre_start_close_price",
        #     "period_start_close_price",
        #     "period_pre_chg_pct",
        #     "period_start_pos_pct",
        #     "hold_days",
        #     "unit_net_change",
        #     "net",
        #     "initial_date",
        #     "initial_price",
        #     "exchange_cd",
        #     "trade_date",
        #     "pre_close_price",
        #     "act_pre_close_price",
        #     "open_price",
        #     "highest_price",
        #     "lowest_price",
        #     "close_price",
        #     "turnover_vol",
        #     "turnover_value",
        #     "deal_amount",
        #     "turnover_rate",
        #     "accum_adj_bf_factor",
        #     "neg_market_value",
        #     "market_value",
        #     "chg_pct",
        #     "pe",
        #     "pe1",
        #     "pb",
        #     "is_open",
        #     "vwap",
        #     "accum_adj_af_factor",
        #     "price_ampl",
        #     "price_ampl_rate",
        #     "ma10_price_ampl_rate",
        #     "ma5_vol_rate",
        #     "list_days",
        #     "wq_alpha16",
        #     "ntra_turnover_rate_5",
        #     "float_rate_60",
        #     "ntra_bias_6",
        #     "ntra_bias_20",
        #     "bias_6",
        #     "chg_pct_21",
        #     "rank",
        #     "neg_market_value_subrank",
        #     "wq_alpha16_subrank",
        #     "ntra_turnover_rate_5_subrank",
        #     "float_rate_60_subrank",
        #     "ntra_bias_6_subrank",
        #     "chg_pct_21_subrank",
        #     "ma10_price_ampl_rate_subrank",
        #     "xrank",
        #     "sale_days",
        # ]
        # self.df_position_mfst = pd.DataFrame(columns=columns_pos_mfst)
        self.df_position_mfst = pd.DataFrame()
        # 历史卖出清单
        # columns_sale_mfst = [
        #     "ticker",
        #     "sec_short_name",
        #     "buy_date",
        #     "sale_date",
        #     "buy_price",
        #     "sale_price",
        #     "chg_pct",
        #     "hold_days",
        #     "unit_net",
        #     "increase_net",
        #     "relative_roi",
        #     "turnover_vol",
        #     "turnover_value",
        #     "deal_amount",
        #     "turnover_rate",
        #     "accum_adj_bf_factor",
        #     "neg_market_value",
        #     "market_value",
        #     "pe",
        #     "pe1",
        #     "pb",
        #     "is_open",
        #     "vwap",
        #     "accum_adj_af_factor",
        #     "price_ampl",
        #     "price_ampl_rate",
        #     "ma10_price_ampl_rate",
        #     "ma5_vol_rate",
        #     "list_days",
        #     "wq_alpha16",
        #     "ntra_turnover_rate_5",
        #     "float_rate_60",
        #     "ntra_bias_6",
        #     "ntra_bias_20",
        #     "bias_6",
        #     "chg_pct_21",
        #     "rank",
        #     "neg_market_value_subrank",
        #     "wq_alpha16_subrank",
        #     "ntra_turnover_rate_5_subrank",
        #     "float_rate_60_subrank",
        #     "ntra_bias_6_subrank",
        #     "chg_pct_21_subrank",
        #     "ma10_price_ampl_rate_subrank",
        #     "xrank",
        #     "unit_net_change",
        # ]
        # self.df_sale_mfst = pd.DataFrame(columns=columns_sale_mfst)
        self.df_sale_mfst = pd.DataFrame()

        # # 过滤出来的股票池
        # self.df_equd_pool = None

        # Strategy Config
        self.config = StrategyConfig()
        self.__mk_folder()
        # 根据添加的指标滚动窗口，记录最大向前滚动的时间窗口 以保证边缘日期的均值计算问题
        self.start_date = None
        self.end_date = None

        self.__load_total_data_set()

        # self.df_equd_pool = self.df_equd_pool
        self.stockutil: StockUtil = StockUtil()

    def set_metric_folder(self, path):
        """For unit test only

        Args:
            path (_type_): _description_
        """
        self.config.data_folder = path
        self.__mk_folder()

    def __mk_folder(self):
        # os.makedirs(f"{self.config.data_folder}/cache/", exist_ok=True)
        os.makedirs(f"{self.config.data_folder}/metrics/", exist_ok=True)

    def __load_total_data_set(self):
        """从数据库加载文件，具备缓存能力, 缓存需要手动删除

        Raises:
            Exception: _description_
            Exception: _description_
        """
        db = DBAdaptor(is_use_cache=True)
        if self.trade_type == 0:
            self.df_equ_pool = db.get_df_by_sql("select * from stock.equity")
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
            df = db.get_df_by_sql(
                f"select * from stock.mkt_equ_day where trade_date >='{self.dataset_start}' and trade_date <= '{self.dataset_end}' order by id"
            )
            # 把量能有关的 停牌日设置为空，预期失真不如排出
            # df.loc[df.open_price==0, 'turnover_vol']= np.nan
            # df.loc[df.open_price==0, 'turnover_rate']= np.nan
            # df.loc[df.open_price==0, 'turnover_value']= np.nan
            # df.loc[df.open_price==0, 'deal_amount']= np.nan
            # 计算价格想逛因子需要考虑停牌日在内，比如涨幅，乖离率
            self.df_equd_pool_with_sus = df
            # 停牌日行情，排出在外
            df = df.loc[df.open_price > 0, :]
            self.df_equd_pool = df
            logger.debug(f"股票池已经设定所有上市股票")
        elif self.trade_type == 1:
            self.df_equd_pool = db.get_df_by_sql(
                f"select * from stock.fund_day  where trade_date >='{self.dataset_start}'  and trade_date <= '{self.dataset_end}' order by id"
            )
            logger.debug(f"股票池已经设定所有ETF基金")
        if self.df_equd_pool.shape[0] == 0:
            raise Exception("从数据库加载日线数据为空!")
        logger.debug(f"股票池已经加载所有日线数据:{self.df_equd_pool.shape[0]}")

    def select_equd_by_equ_pool(self):
        """设置完自选股池后，需要手动调用这个函数更新 选股日线行情池

        Returns:
            _type_: _description_
        """
        df = self.df_equd_pool
        oc = df.shape[0]
        df = df[df["ticker"].isin(self.df_equ_pool["ticker"])]
        self.df_equd_pool = df
        nc = self.df_equd_pool.shape[0]
        logger.debug(f"K线数据已根据股票池更新{self.df_equd_pool.shape[0]}, 过滤掉{nc-oc}")
        return df

    def append_select_condition(self, condition):
        """添加每日选股筛选条件

        Args:
            condition (_type_): _description_
        """
        # if self.debug_sample_date is not None:
        #     oc = self.df_equd_pool[
        #         self.df_equd_pool.trade_date == pd.to_datetime(self.debug_sample_date)
        #     ].shape[0]

        # self.df_equd_pool = self.df_equd_pool.query(condition)
        if self.select_conditions:
            self.select_conditions = (
                self.select_conditions + " and " + condition
            )
        else:
            self.select_conditions = condition

        # if self.debug_sample_date is not None:
        #     nc = self.df_equd_pool[
        #         self.df_equd_pool.trade_date == pd.to_datetime(self.debug_sample_date)
        #     ].shape[0]

        # self.select_conditions.append(condition)
        # if self.debug_sample_date is not None:
        #     logger.debug(
        #         f"已加载条件{condition}, {self.debug_sample_date} 现存:{nc} 过滤掉: {nc-oc}个股票"
        # )

    def append_metric(self, sm: SelectMetric, reset_cache: bool = False):
        """增加指标

        Args:
            sm (SelectMetric): _description_
            reset_cache (bool, optional): _description_. Defaults to False.
        """
        self.df_equd_pool[sm.name] = self.__add_metric_column(sm, reset_cache)
        self.df_metrics.append(sm.name)

        # Debug information
        if self.debug_sample_date is not None:
            df = self.df_equd_pool.loc[
                self.df_equd_pool.trade_date
                == pd.to_datetime(self.debug_sample_date),
                ["sec_short_name", sm.name],
            ]
            logger.debug(f"指标{sm.name}已加载, {self.debug_sample_date} 指标值: {df}")

    def append_metrics(self, sms: List[SelectMetric]):
        for sm in sms:
            self.append_metric(sm)

    def __add_metric_column(
        self, sm: SelectMetric, reset_cache: bool = False
    ) -> pd.Series:
        metric_cache_file = (
            f"{self.config.data_folder}/metrics/{sm.name}.paquet"
        )

        if reset_cache or not os.path.exists(metric_cache_file):
            # Compute the metric
            df_metric = sm.apply(self.df_equd_pool, sm.name, sm.args)
            df_metric.to_parquet(metric_cache_file, index=True)
        else:
            # Load from file
            df_metric = pd.read_parquet(metric_cache_file)

        logger.debug(f"指标{sm.name}已经加载到df_equd_pool 列表中")
        return df_metric[sm.name]

    def post_hook_select_equ(self):
        raise Exception("这个方法是抽象方法，应该在子类中实现")
        pass

    def append_rankfactor(self, rf: RankFactor):
        """增加每日选股的排序

        Args:
            rf (RankFactor): _description_

        Raises:
            Exception: _description_
        """
        cols = self.df_equd_pool.columns

        if rf.name not in cols:
            raise Exception(f"排序指标{rf.name}没有在df_equd_pool列中找到:{cols}")

        if self.rank_factors is None:
            self.rank_factors = list()
        self.rank_factors.append(rf)

    def append_rankfactors(self, rfs: List[RankFactor]):
        if self.rank_factors is None:
            self.rank_factors = list()

        self.rank_factors.extend(rfs)

    def rank(self) -> pd.DataFrame:
        """对每日选股中的自选股列表进行排序
        注意它包含历史所有日期到的数据 ，需要使用groupby

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """

        df: pd.DataFrame = self.df_equd_pool
        df["rank"] = 0
        tw = 0
        for rf in self.rank_factors:
            if rf.name not in df.columns:
                raise Exception(
                    f"排序因子{rf.name}无法在df_equd_pool的列中找到:{df.columns}"
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
        df["rank"] = df.groupby("trade_date")["xrank"].transform(
            "rank",
            ascending=False,
        )

        df.sort_values(["trade_date", "xrank"], ascending=False, inplace=True)

        self.df_equd_pool = df

        if self.debug_sample_date is not None:
            df = self.df_equd_pool.loc[
                self.df_equd_pool.trade_date
                == pd.to_datetime(self.debug_sample_date),
                :,
            ]
            logger.debug(f"排序已完成, {self.debug_sample_date}  指标值: {df}")
        logger.debug(f"选好的股票已经排序")
        return df

    def rank_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """对给定的dataframe 里面的元素进行排序, 不是针对全量日期, 是针对一天的

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """
        starttime = datetime.now()

        df["rank"] = 0
        tw = 0
        for rf in self.rank_factors:
            if rf.name not in df.columns:
                raise Exception(
                    f"排序因子{rf.name}无法在df_equd_pool的列中找到:{df.columns}"
                )
            subrank_column = rf.name + "_subrank"
            df[subrank_column] = (
                df[rf.name].transform(
                    "rank", ascending=rf.bigfirst, pct=True, method="max"
                )
                * 100
            )
            df["rank"] = df["rank"] + df[subrank_column] * rf.weight
            tw = tw + rf.weight

        df["rank"] = df["rank"] / tw

        # 转换为百分制
        df["xrank"] = (
            df["rank"].transform(
                "rank", ascending=True, pct=True, method="max"
            )
            * 100
        )
        df["rank"] = df["xrank"].transform(
            "rank",
            ascending=False,
        )

        df.sort_values(["xrank"], ascending=False, inplace=True)

        endtime = datetime.now()
        logger.debug(
            f"选好的股票已重新排序完成, 花费时间:{(endtime-starttime).total_seconds()*1000}毫秒"
        )
        return df

    def select_equd_by_daterange(
        self, start_date: date, end_date: date = None
    ) -> pd.DataFrame:

        df = self.df_equd_pool
        self.start_date = start_date

        if end_date is None:
            df = df[(df.trade_date >= pd.to_datetime(start_date))]
            self.end_date = self.dataset_end
        else:
            self.end_date = end_date
            df = df[
                (df.trade_date >= pd.to_datetime(start_date))
                & (df.trade_date <= pd.to_datetime(end_date))
            ]
        self.df_equd_pool = df

        return df

    def generate_position_mfst(self) -> pd.DataFrame:
        """使用交易模型产生交易清单

        Args:
            start_date (date): _description_
            end_date (date, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        starttime_0 = datetime.now()
        tm: TradeModel = self.trade_model
        # 11 columns
        position_mfst_columns = [
            "period",
            "initial_date",
            "initial_price",
            "start_date",
            "end_date",
            "ticker",
            "period_pre_start_close_price",
            "period_start_pos_pct",
            "period_pre_chg_pct",
            "hold_days",
            "net",
        ]
        sale_mfst_columns = [
            "ticker",
            "sec_short_name",
            "buy_date",
            "sale_date",
            "buy_price",
            "sale_price",
            "chg_pct",
            "hold_days",
            "unit_net",
            "increase_net",
            "relative_roi",
        ]

        # 8个变量进入循环
        pre = pd.DataFrame(columns=position_mfst_columns)
        cur = None
        start_date = self.start_date
        end_date = None
        period = 1
        # 余额
        balance = 1
        # 剩余可购买仓位
        # 当前净值
        cur_net = 1
        # 当日自选股行情
        cur_choice_equd = None

        while start_date <= self.end_date:
            starttime = datetime.now()
            end_date = self.stockutil.get_trade_date_by_offset(
                start_date, int(-1 * tm.xperiod)
            )

            cur_choice_equd = self.get_daily_choice_equd(start_date, pre)
            if cur_choice_equd.shape[0]==0:
                logger.warning(f"period:{period} {start_date} 选股为空，没有选到股票!")
            # 先判断卖出
            if pre.shape[0] > 0:
                pre["period"] = period
                pre["start_date"] = start_date
                pre["end_date"] = end_date
                pre["hold_days"] = pre["hold_days"] + tm.xperiod
                pre["period_pre_start_close_price"] = pre["close_price"]

                # 判断是否出现不在自选股K线池里面
                # 当日自选股
                # cur_choice_equd = self.df_equd_pool.loc[
                #     self.df_equd_pool.trade_date == pd.to_datetime(start_date), :
                # ]
                # not_in_choice_equd = pre.loc[
                #     (~pre.ticker.isin(cur_choice_equd.ticker)), :
                # ]
                # not_in_choice_equd_appear = (
                #     not_in_choice_equd.shape[0] >0
                # )

                # if not_in_choice_equd_appear:
                #     starttime_1 = datetime.now()
                #     # 取得当日自选股的equd行情(带指标），然后加入持仓股进入自选股
                #     # 需要重新排序
                #     cur_equd_pool = self.df_equd_pool.loc[(self.df_equd_pool.trade_date == pd.to_datetime(start_date)),:]
                #     not_in_choice_equd = cur_equd_pool.loc[
                #         (cur_equd_pool.ticker.isin(not_in_choice_equd.ticker)),
                #         :,
                #     ]
                #     endtime_1 = datetime.now()
                #     logger.debug(f'查询outlier股票 {(endtime_1-starttime_1).total_seconds()*1000} 毫秒')
                #     if not_in_choice_equd.shape[0] > 0:
                #         logger.debug(
                #             f"出现不在自选股K线池, 但是已经持仓的股票里面{not_in_choice_equd.trade_date} {not_in_choice_equd.sec_short_name}"
                #         )

                #         # 加入当日的自选股，因为已经持有
                #         cur_choice_equd = pd.concat(
                #             [cur_choice_equd, not_in_choice_equd]
                #         )
                #         # 对当日重新排名
                #         cur_choice_equd = self.rank_df(cur_choice_equd)

                update_pre_equd = cur_choice_equd.loc[
                    cur_choice_equd.ticker.isin(pre.ticker),
                    cur_choice_equd.columns[2 : len(cur_choice_equd.columns)],
                ]

                pre.drop(
                    pre.columns[11 : len(pre.columns)], axis=1, inplace=True
                )

                pre = pd.merge(pre, update_pre_equd, on="ticker", how="left")

                if pre.shape[0] > update_pre_equd.shape[0]:
                    # 出现了停牌的股票, 需要特殊处理
                    suspend_pos = pre[~pre.ticker.isin(update_pre_equd.ticker)]
                    logger.debug(
                        f"出现停牌股票，{suspend_pos[['start_date','ticker']]}"
                    )

                    # 手动设置close_price 而不能是0 因为要计算净值
                    # 停牌日收盘价本来就不是空, 如果为空，说明数据有问题
                    # pre.loc[
                    #     (pre.open_price.isna()) | (pre.open_price == 0),
                    #     "close_price",
                    # ] = pre["period_pre_start_close_price"]
                    # pre.loc[
                    #     (pre.open_price.isna()) | (pre.open_price == 0),
                    #     "sec_short_name",
                    # ] = "当日停牌"
                    # pre.loc[pre.open_price.isna(), "open_price"] = 0

                # 统计上个周期的涨跌幅
                pre["period_pre_chg_pct"] = (
                    pre["close_price"] / pre["period_pre_start_close_price"]
                    - 1
                )

                pre_net: float = pre["net"].iloc[0]
                pre.loc[:, "unit_net_change"] = (
                    pre["period_pre_chg_pct"]
                    * pre["period_start_pos_pct"]
                    * pre_net
                )
                cur_net: float = (
                    (pre["close_price"] / pre["period_pre_start_close_price"])
                    * pre["period_start_pos_pct"]
                    * pre_net
                ).sum() + balance
                # pre["net"] = cur_net
                # pre["period_start_pos_pct"] = (
                #     pre["period_start_pos_pct"]
                #     * pre_net
                #     * (
                #         pre["close_price"]
                #         / pre["period_pre_start_close_price"]
                #     )
                # ) / cur_net

                # Check Sale condition
                sale = pre.query(tm.sale_criterial)

                if tm.notsale_criterial is not None:
                    notsale = sale.query(tm.notsale_criterial)

                    if notsale is not None and notsale.shape[0] > 0:
                        sale = sale[~sale.ticker.isin(notsale.ticker)]

                # 空出仓位值
                sum_sale_trans_fee = (
                    sale["period_start_pos_pct"]
                    * tm.sale_fee_rate
                    * pre_net
                    * (1 + pre["period_pre_chg_pct"])
                ).sum()
                balance: float = (
                    (
                        sale["period_start_pos_pct"]
                        * pre_net
                        * (1 + pre["period_pre_chg_pct"])
                    ).sum()
                    + balance
                    - sum_sale_trans_fee
                )

                pre = pre[~pre.ticker.isin(sale.ticker)]
                # if notsale is not None and notsale.shape[0] > 0:
                #     pre = pd.concat([pre, notsale])

                # # 检查是否有超出边界 但是没有到达卖出条件的股票
                # max_allow_pos_pct = tm.unit_ideal_pos_pct * (
                #     1 + tm.unit_pos_pct_tolerance
                # )
                # sale_partial = pre.query(
                #     f"period_start_pos_pct>{max_allow_pos_pct}"
                # )

                # if sale_partial.shape[0] > 0:
                #     partial_sum_sale_trans_fee = (
                #         (
                #             sale["period_start_pos_pct"]
                #             - tm.unit_ideal_pos_pct
                #         )  # 超出的仓位
                #         * cur_net
                #         * tm.sale_fee_rate
                #     ).sum()
                #     balance_sale_partial = (
                #         (
                #             sale["period_start_pos_pct"]
                #             - tm.unit_ideal_pos_pct
                #         )  # 超出的仓位
                #         * cur_net
                #     ).sum() - partial_sum_sale_trans_fee
                #     balance = balance_sale_partial + balance
                #     cur_net = cur_net - partial_sum_sale_trans_fee
                #     self.df_sale_mfst = pd.concat(
                #         [self.df_sale_mfst, sale_partial]
                #     )
                #     # 超出仓位的股票回归理想仓位
                #     pre.loc[
                #         (
                #             pre["period_start_pos_pct"]
                #             > tm.unit_ideal_pos_pct * tm.unit_pos_pct_tolerance
                #         ),
                #         "period_start_pos_pct",
                #     ] = tm.unit_ideal_pos_pct

                cur_net = cur_net - sum_sale_trans_fee
                pre.loc[:, "net"] = cur_net

                pre.loc[:, "period_start_pos_pct"] = (
                    pre["period_start_pos_pct"]
                    * pre_net
                    * (
                        pre["close_price"]
                        / pre["period_pre_start_close_price"]
                    )
                ) / cur_net
                # 记录卖出股票
                sale_mfst = pd.DataFrame(columns=sale_mfst_columns)
                sale_mfst["ticker"] = sale["ticker"]
                sale_mfst["sec_short_name"] = sale["sec_short_name"]
                sale_mfst["buy_date"] = sale["initial_date"]
                sale_mfst["buy_price"] = sale["initial_price"]
                sale_mfst["sale_date"] = sale["trade_date"]
                sale_mfst["sale_price"] = sale["close_price"]
                sale_mfst["chg_pct"] = (
                    sale_mfst["sale_price"] / sale_mfst["buy_price"] - 1
                )
                sale_mfst["hold_days"] = sale["hold_days"]

                # This is incorrect
                # sale_mfst["unit_net"] = (
                #     sale["period_start_pos_pct"] * sale["net"]
                # )
                # sale_mfst["increase_net"] = (
                #     sale["unit_net"] * sale["chg_pct"] * (1-tm.sale_fee_rate)/ (1 + sale["chg_pct"])
                # )
                sale_mfst["relative_roi"] = 0
                # Keep the metrics for analysis
                sale.drop(sale.columns[0:20], axis=1, inplace=True)
                sale.drop(labels=["chg_pct"], axis=1, inplace=True)
                sale_mfst = sale_mfst.join(sale)
                # sale_mfst["relative_roi"] = sale_mfst["chg_pct"] + 1

                self.df_sale_mfst = pd.concat([self.df_sale_mfst, sale_mfst])

                # 上次持有，排除掉已经清仓的，过渡到，当前一期
                cur = pre
            else:
                cur = pd.DataFrame(columns=position_mfst_columns)
                pass
            # 买入新的股票
            # 检查目前仓位是否有低于容忍百分比的股票
            # 暂时不实现
            # buy_partial = cur.query(
            #     "'period_start_pos_pct'<tm.unit_ideal_pos_pct * (1-tm.unit_pos_pct_tolerance)"

            # )
            # if buy_partial is not None and buy_partial.shape[0]>0:
            #     # 补仓

            # cur_choice_equd = self.get_choice_equ_by_date(start_date)
            # if cur_choice_equd is None or cur_choice_equd.shape[0] == 0:
            #     cur_choice_equd = self.get_daily_choice_equd(start_date)
            # logger.debug(
            #     f"周期:{period} {start_date}选股{cur_choice_equd.shape[0]}"
            # )
            # 此处使用.loc[:, "sale_days"]会有bug，如果cur_choice_equd 为空就会出错
            cur_choice_equd["sale_days"] = 100000
            # 增加卖出时间列
            # 增加临时卖出天数列 , 0 表示没有卖过
            if (
                self.df_sale_mfst.shape[0] > 0
                and self.df_sale_mfst.ticker.isin(cur_choice_equd.ticker).any()
            ):
                saled_ticker = self.df_sale_mfst[["ticker", "sale_date"]]
                saled_ticker.drop_duplicates(
                    subset="ticker", keep="last", inplace=True
                )

                cur_choice_equd = pd.merge(
                    cur_choice_equd, saled_ticker, on="ticker", how="left"
                )
                cur_choice_equd["sale_days"] = (
                    cur_choice_equd["trade_date"]
                    - cur_choice_equd["sale_date"]
                ).dt.days
                cur_choice_equd.drop(["sale_date"], axis=1, inplace=True)

            cur_choice_equd["sale_days"].fillna(100000, inplace=True)

            df_buy_candidate = cur_choice_equd.query(tm.buy_criterial)

            df_buy_candidate = df_buy_candidate[
                ~df_buy_candidate.ticker.isin(cur.ticker)
            ]

            buy_candidate_count = df_buy_candidate.shape[0]

            # 调试作用
            if balance - cur_net > 0.001:
                error_position_file = "/tmp/error_df_position_mfst.csv"
                error_sale_file = "/tmp/error_df_sale_mfst.csv"
                self.df_position_mfst.to_csv(error_position_file)
                self.df_sale_mfst.to_csv(error_sale_file)
                logger.error(
                    f"内部错误, 请检查{error_position_file} {error_sale_file}"
                )
                raise Exception("余额不应该大于总的净值")

            # 计算最大可买股票数量
            balance_pos_pct = balance / cur_net
            max_buy_count = int(balance_pos_pct / tm.unit_ideal_pos_pct)
            if (balance_pos_pct / tm.unit_ideal_pos_pct).is_integer():
                is_dividable = True
                nondividable_balance_pos_change = 0
            else:
                nondividable_balance_pos_change = (
                    balance_pos_pct - max_buy_count * tm.unit_ideal_pos_pct
                )
                if nondividable_balance_pos_change >= tm.mini_unit_buy_pct:
                    max_buy_count = max_buy_count + 1
                is_dividable = False

            if buy_candidate_count > 0:
                # 已选出股票了
                if max_buy_count > buy_candidate_count:
                    # 因为候选股票不足 导致买不足
                    buy_count = buy_candidate_count
                    all_in = False
                else:
                    all_in = True
                    buy_count = max_buy_count

                df_buy_candidate = df_buy_candidate[0:buy_count]
                # 购买的时候只要设置7个列，剩余两个列一个是ticker，另一个是pre_close_price
                df_buy_candidate["period"] = period
                df_buy_candidate["start_date"] = start_date
                df_buy_candidate["end_date"] = end_date

                df_buy_candidate["period_pre_chg_pct"] = 0
                df_buy_candidate["hold_days"] = 0
                df_buy_candidate["net"] = cur_net

                # 创建两个临时列建仓日期
                df_buy_candidate["initial_date"] = start_date
                df_buy_candidate["initial_price"] = df_buy_candidate[
                    "close_price"
                ]

                df_buy_candidate[
                    "period_start_pos_pct"
                ] = tm.unit_ideal_pos_pct
                if all_in:
                    # 调整最后一个股票的仓位
                    if not is_dividable:
                        if (
                            nondividable_balance_pos_change
                            >= tm.mini_unit_buy_pct
                        ):
                            balance_pos_pct = (
                                balance_pos_pct
                                - (buy_count - 1) * tm.unit_ideal_pos_pct
                            )
                            df_buy_candidate.loc[
                                :, "period_start_pos_pct"
                            ].iloc[-1] = balance_pos_pct
                            balance = 0
                        else:
                            balance = nondividable_balance_pos_change * cur_net
                    else:
                        balance = 0
                else:
                    balance = (
                        balance - buy_count * tm.unit_ideal_pos_pct * cur_net
                    )
                # df_buy_candidate["unit_net_change"] = 0
                cur = pd.concat(
                    [
                        cur,
                        df_buy_candidate[
                            df_buy_candidate.columns[
                                2 : len(df_buy_candidate.columns)
                            ]
                        ],
                    ]
                )
            else:
                # 没有选出股票

                pass

            if cur.shape[0] > 0:
                self.df_position_mfst = pd.concat([self.df_position_mfst, cur])
                pre = cur.copy()
            else:
                pre = pd.DataFrame(columns=position_mfst_columns)
                pass

            # 调试作用
            if cur["period_start_pos_pct"].sum() > 1.001:
                debug_file_position_mfst = "/tmp/error_df_position_mfst.csv"
                debug_file_sale_mfst = "/tmp/error_df_sale_mfst.csv"
                self.df_position_mfst.to_csv(debug_file_position_mfst)
                self.df_sale_mfst.to_csv(debug_file_sale_mfst)
                raise Exception(
                    f"内部错误, 持仓股票超出100%比例，请检查{debug_file_position_mfst} {debug_file_sale_mfst}"
                )
            period = period + 1
            start_date = end_date
            endtime = datetime.now()
            logger.debug(
                f"计算一日回测花费时间{(endtime-starttime).total_seconds()*1000} 毫秒"
            )

        # 计算历史最大回撤
        if self.df_position_mfst.shape[0] == 0:
            logger.warning("{self.start_date}-{self.end_date},没有持仓的股票")
        else:
            max_net = self.df_position_mfst["net"].cummax()
            self.df_position_mfst["drawback_pct"] = (
                self.df_position_mfst["net"] / max_net - 1
            )
            # self.df_position_mfst['rolling_on_year_max_net'] = self.df_position_mfst['net'].rolling(243, min_periods=1).max
            # self.df_position_mfst['drawback_pct'] = self.df_position_mfst['net']/max_net -1

        endtime_0 = datetime.now()
        logger.debug(
            f"计算历史回测花费时间{(endtime_0-starttime_0).total_seconds()*1000} 毫秒"
        )
        return self.df_position_mfst

    def get_fmt_position_mfst(
        self, start_date: date = None, end_date: date = None
    ):
        """美化持仓清单增强可读性

        Args:
            start_date (date, optional): _description_. Defaults to None.
            end_date (date, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        df = self.df_position_mfst
        # 增加一列
        df["period_start_close_price"] = df["close_price"]
        # 调整列的顺序 增强可读性
        reorder_pos_mfst_columns = [
            "period",
            "sec_short_name",
            "ticker",
            "start_date",
            "end_date",
            "period_pre_start_close_price",
            "period_start_close_price",
            "period_pre_chg_pct",
            "period_start_pos_pct",
            "hold_days",
            "unit_net_change",
            "net",
            "initial_date",
            "initial_price",
        ]
        # 仍然将原来的买入时机的股票指标显示
        reorder_pos_mfst_columns.extend(df.columns[12 : len(df.columns) - 3])
        df = df[reorder_pos_mfst_columns]
        return df

    def get_history_max_drawdown(self) -> float:
        """获取历史最大回撤

        Returns:
            float: _description_
        """
        if self.df_position_mfst.shape[0] == 0:
            logger.warning("没有发现持仓股票")
            max_drawback = 0
        else:
            max_drawback = self.df_position_mfst["drawback_pct"].min() * 100
        return max_drawback

    def get_history_max_roi(self) -> float:
        """获取历史最大收益

        Returns:
            float: _description_
        """
        ret = self.df_position_mfst["net"].max()
        return ret

    def get_roi_by_date(self) -> float:
        """获取最终的收益倍数

        Args:
            start_date (date): _description_
            end_date (date): _description_

        Returns:
            float: _description_
        """
        ret = self.df_position_mfst["net"].iloc[-1]
        return ret

    def get_anual_roi(self) -> float:
        """获取平均年化收益

        Returns:
            float: _description_
        """
        start_date = self.start_date
        end_date = self.end_date
        final_roi = self.get_roi_by_date()
        annu_roi = final_roi ** (1 / ((end_date - start_date).days / 365)) - 1
        return annu_roi
        # df= df_position_mfst
        # df['year'] = df['start_date'].dt.to_period['Y']
        # def cac_in_year_chg(x):
        #     begin_net = x['net'].iloc[0]
        #     end_net = x['net'].iloc[-1]
        #     x['in_year_chg'] = (x['net']-begin_net)/begin_net
        #     x['end_year_chg'] = (end_net-begin_net)/begin_net

        # df = df.groupby('year').apply(cac_in_year_chg)
        # df_annu = df[['year','end_year_chg']].drop_duplicates(subset=['year'])
        # ann_return = df_annu['end_year_chg']

        # N = len(ann_return)
        # ret = ann_return.add(1).prod() ** (12 / N) - 1

    def get_sale_mfst_by_date(
        self,
        trade_date: date,
    ) -> pd.DataFrame:
        """产生交易清单

        Args:
            start_date (date): _description_
            end_date (date, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        mfst = pd.DataFrame()
        # df = self.df_equd_pool

        return mfst

    def get_choice_equ_by_date(self, trade_date: date) -> pd.DataFrame:
        """获取某日的选股

        Args:
            trade_date (date): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = self.df_equd_pool
        return df.loc[df.trade_date == pd.to_datetime(trade_date), :]

    def get_daily_choice_equd(
        self, trade_date: date, pos: pd.DataFrame = None
    ) -> pd.DataFrame:
        """获取某日的选股, 并且要包含每日持仓股

        Args:
            trade_date (date): _description_

        Returns:
            pd.DataFrame: _description_
        """
        starttime = datetime.now()
        if self.equd_groupby_date is None:
            self.equd_groupby_date = self.df_equd_pool.groupby("trade_date")
        df = self.equd_groupby_date.get_group(trade_date)
        # 筛选
        df = df.query(self.select_conditions)
        if pos is not None:
            outlier = pos.loc[~pos.ticker.isin(df.ticker), :]
            if outlier.shape[0] > 0:
                logger.debug(f"已把跳出自选股的持仓中的股票加入到自选股中 {outlier}")
                df = pd.concat([df, outlier[df.columns[2 : len(df.columns)]]])

        # 排序
        df = self.rank_df(df)
        endtime = datetime.now()
        logger.debug(
            f"每日实时选股+ 排序完成, 花费时间:{(endtime-starttime).total_seconds()*1000}毫秒"
        )
        return df

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

    def get_choice_equ_metrics_by_list(
        self, choice: pd.DataFrame
    ) -> pd.DataFrame:
        """根据选股列表，反向获取所有指标的值

        Args:
            choice (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = self.df_equd_pool
        df.trade_date = pd.to_datetime(df.trade_date)
        pool_count=pd.DataFrame()
        pool_count['select_equ_count']=df.groupby('trade_date').size()
        # pool_count['trade_date']=pool_count.index

        # choice.ticker=choice.ticker.astype('string')
        choice.trade_date = pd.to_datetime(choice.trade_date)
        choice = pd.DataFrame.merge(
            choice, df, on=["ticker", "trade_date"], how="left"
        )
        choice=pd.DataFrame.merge(
            choice, pool_count, on=["trade_date"], how="left"
        )
        return choice

    def reset_backtest_result(self):
        self.df_position_mfst = pd.DataFrame()
        self.df_sale_mfst = pd.DataFrame()
