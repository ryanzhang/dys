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
from datetime import date, timedelta
from typing import Any, Iterable, List

import pandas as pd
from kupy.config import configs
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.domain import RankFactor, SelectMetric, StrategyConfig, TradeModel
from dys.stockutil import StockUtil

# By default 120 tradeable days between 2009-07-09 to 2020-12-31
# New k data can be append since 2022-01-04
DEFAULT_TOTAL_START_DATE = "20090709"
DEFAULT_TOTAL_END_DATE = "20220223"


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

        self.select_conditions = None
        self.debug_sample_date = None
        self.trade_model = None
        self.rank_factors = None
        # self.select_metrics = None
        self.df_metrics: List(str) = list()

        self.mkt_timing_algo = None
        # 历史持仓
        self.df_position_mfst = pd.DataFrame()
        # 历史卖出清单
        self.df_sale_mfst = pd.DataFrame()

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
        self.stockutil: StockUtil = StockUtil()

    def __set_cache_folder(self, path):
        self.config.data_folder = path
        self.__mk_folder()

    def __mk_folder(self):
        os.makedirs(self.config.data_folder + "cache/", exist_ok=True)
        os.makedirs(self.config.data_folder + "metrics/", exist_ok=True)

    def __load_total_data_set(self):
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

    def append_metric(self, sm: SelectMetric, reset_cache: bool = False):
        self.df_choice_equd[sm.name] = self.__add_metric_column(
            sm, reset_cache
        )
        self.df_metrics.append(sm.name)

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

    def __add_metric_column(
        self, sm: SelectMetric, reset_cache: bool = False
    ) -> pd.Series:
        metric_cache_file = (
            f"{self.config.data_folder}/metrics/{sm.name}.paquet"
        )

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
        df["rank"] = df.groupby("trade_date")["xrank"].transform(
            "rank",
            ascending=False,
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
        self.start_date = start_date

        if end_date is None:
            df = df[(df.trade_date >= start_date)]
            self.end_date = self.dataset_end
        else:
            self.end_date = end_date
            df = df[
                (df.trade_date >= start_date) & (df.trade_date <= end_date)
            ]
        self.df_choice_equd = df

        return df

    def generate_position_mfst(self) -> pd.DataFrame:
        """产生交易清单

        Args:
            start_date (date): _description_
            end_date (date, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
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

        while start_date <= self.end_date:
            end_date = self.stockutil.get_trade_date_by_offset(
                start_date, -1 * tm.xperiod
            )

            # 先判断卖出
            if pre.shape[0] > 0:
                pre["period"] = period
                pre["start_date"] = start_date
                pre["end_date"] = end_date
                pre["hold_days"] = pre["hold_days"] + tm.xperiod
                pre["period_pre_start_close_price"] = pre["close_price"]

                update_pre_equd = self.df_choice_equd.loc[
                    (self.df_choice_equd.ticker.isin(pre.ticker))
                    & (self.df_choice_equd.trade_date == start_date),
                    self.df_choice_equd.columns[
                        2 : len(self.df_choice_equd.columns)
                    ],
                ]

                # 是否出现停牌股票
                not_in_choice_appear = False
                if pre.shape[0] > update_pre_equd.shape[0]:
                    # 出现了停牌的股票, 需要特殊处理
                    not_in_choice_appear = True
                    not_in_choice_equ = pre[~pre.ticker.isin(update_pre_equd.ticker)]
                    logger.debug(f'出现ST股票，已经无法在指标选股列表中找到行情数据{not_in_choice_equ}')
                    not_in_choice_equ["close_price"] = not_in_choice_equ[
                        "period_pre_start_close_price"
                    ]
                    not_in_choice_equ["open_price"] = 0
                    not_in_choice_equ["highest_price"] = 0
                    not_in_choice_equ["lowest_price"] = 0

                    # pre = pre[pre.ticker.isin(update_pre_equd.ticker)]

                pre.drop(
                    pre.columns[11 : len(pre.columns)], axis=1, inplace=True
                )

                if not_in_choice_appear:
                    # 直接从最大的股票池里面加载数据 并且强制表示为*ST
                    update_not_in_choice_equ = self.df_equd_pool.loc[
                        (self.df_equd_pool.ticker.isin(not_in_choice_equ.ticker))
                        & (self.df_equd_pool.trade_date == start_date),
                        self.df_equd_pool.columns[
                            2 : len(self.df_equd_pool.columns)
                        ],
                    ]
                    # 如果 因为什么原因没有表*ST，我们强制标*ST 以此做为标记要卖出
                    update_not_in_choice_equ.loc[~update_not_in_choice_equ['sec_short_name'].str.startswith('*ST'),'sec_short_name'] = "*ST" + update_not_in_choice_equ['sec_short_name']
                    update_pre_equd = pd.concat([update_pre_equd, update_not_in_choice_equ])
                    # pre = pd.merge(pre, update_not_in_choice_equ, on="ticker", how="left")

                pre = pd.merge(pre,update_pre_equd , on="ticker", how="left")
                    # not_in_choice_equ = pd.merge(not_in_choice_equ, update_not_in_choice_equ, on='ticker', how = 'left')
                    # sale_nice = not_in_choice_equ[not_in_choice_equ.open_price>0]

                    # if sale_nice.shape[0]>0:
                    #     # 空出仓位值
                    #     balance: float = (
                    #         sale_nice["period_start_pos_pct"]
                    #         * sale_nice["net"]
                    #         * (sale_nice['close_price']/sale_nice['period_pre_start_close_price'] -1)
                    #         * (1 - tm.sale_fee_rate)
                    #     ).sum() + balance                        
                    # # 被迫持有停盘股票，等待下一个周期 检查
                    # cant_sale_nice = not_in_choice_equ[not_in_choice_equ.open_price==0]
                    # pre = pd.merge(pre, cant_sale_nice, on="ticker", how="left")

                # update_not_in_choice_equ = update_not_in_choice_equ[update_not_in_choice_equ.open_price>0]


                pre["period_pre_chg_pct"] = (pre["period_pre_chg_pct"] + 1) * (
                    pre["close_price"] / pre["period_pre_start_close_price"]
                ) - 1

                pre_net: float = pre["net"].iloc[0]
                cur_net: float = (
                    (pre["close_price"] / pre["period_pre_start_close_price"])
                    * pre["period_start_pos_pct"]
                    * pre_net
                ).sum() + balance
                pre["net"] = cur_net
                pre["period_start_pos_pct"] = (
                    pre["period_start_pos_pct"]
                    * pre_net
                    * (
                        pre["close_price"]
                        / pre["period_pre_start_close_price"]
                    )
                ) / cur_net

                # Check Sale condition
                sale = pre.query(tm.sale_criterial)

                notsale = sale.query(tm.notsale_criterial)

                if notsale is not None and notsale.shape[0] > 0:
                    sale = sale[~sale.ticker.isin(notsale.ticker)]

                # 空出仓位值
                balance: float = (
                    sale["period_start_pos_pct"]
                    * (1 - tm.sale_fee_rate)
                    * sale["net"]
                ).sum() + balance

                pre = pre[~pre.ticker.isin(sale.ticker)]
                if notsale is not None and notsale.shape[0] > 0:
                    pre = pd.concat([pre, notsale])

                # 检查是否有超出边界 但是没有到达卖出条件的股票
                max_allow_pos_pct = tm.unit_ideal_pos_pct * (
                    1 + tm.unit_pos_pct_tolerance
                )
                sale_partial = pre.query(
                    f"period_start_pos_pct>{max_allow_pos_pct}"
                )

                if sale_partial.shape[0] > 0:
                    balance_sale_partial = (
                        (
                            sale["period_start_pos_pct"]
                            - tm.unit_ideal_pos_pct
                        )  # 超出的仓位
                        * sale["net"]
                    ).sum()
                    balance = balance_sale_partial + balance
                    self.df_sale_mfst = pd.concat(
                        [self.df_sale_mfst, sale_partial]
                    )
                    # 超出仓位的股票回归理想仓位
                    pre.loc[
                        (
                            pre["period_start_pos_pct"]
                            > tm.unit_ideal_pos_pct * tm.unit_pos_pct_tolerance
                        ),
                        "period_start_pos_pct",
                    ] = tm.unit_ideal_pos_pct

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
                sale_mfst["unit_net"] = (
                    sale["period_start_pos_pct"] * sale["net"]
                )
                sale_mfst["increase_net"] = (
                    sale["net"] * sale["chg_pct"] / (1 + sale["chg_pct"])
                )
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

            df_candidate = self.get_choice_equ_by_date(start_date)

            df_candidate['sale_days'] = 100000
            # 增加卖出时间列
            # 增加临时卖出天数列 , 0 表示没有卖过
            if (
                self.df_sale_mfst.shape[0] > 0
                and self.df_sale_mfst.ticker.isin(df_candidate.ticker).any()
            ):
                saled_ticker = self.df_sale_mfst[["ticker", "sale_date"]]
                saled_ticker.drop_duplicates(
                    subset="ticker", keep="last", inplace=True
                )

                df_candidate = pd.merge(
                    df_candidate, saled_ticker, on="ticker", how="left"
                )
                df_candidate["sale_days"] = (
                    df_candidate["trade_date"] - df_candidate["sale_date"]
                ).dt.days
                df_candidate.drop(['sale_date'], axis=1, inplace=True)

            df_candidate['sale_days'].fillna(100000, inplace=True)

            df_buy_candidate = df_candidate.query(tm.buy_criterial)

            df_buy_candidate = df_buy_candidate[
                ~df_buy_candidate.ticker.isin(cur.ticker)
            ]

            buy_candidate_count = df_buy_candidate.shape[0]

            # 调试作用
            if balance > cur_net:
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
                            df_buy_candidate["period_start_pos_pct"].iloc[
                                -1
                            ] = balance_pos_pct
                            balance = 0
                        else:
                            balance = nondividable_balance_pos_change * cur_net
                    else:
                        balance = 0
                else:
                    balance = (
                        balance - buy_count * tm.unit_ideal_pos_pct * cur_net
                    )

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

            period = period + 1
            start_date = end_date
            
        # 计算历史最大回撤
        max_net = self.df_position_mfst['net'].cummax()
        self.df_position_mfst['drawback_pct'] = self.df_position_mfst['net']/max_net -1
        # self.df_position_mfst['rolling_on_year_max_net'] = self.df_position_mfst['net'].rolling(243, min_periods=1).max
        # self.df_position_mfst['drawback_pct'] = self.df_position_mfst['net']/max_net -1

        return self.df_position_mfst

    def get_fmt_position_mfst(
        self, start_date: date = None, end_date: date = None
    ):
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
            "net",
            "initial_date",
            "initial_price",
        ]
        # 仍然将原来的买入时机的股票指标显示
        reorder_pos_mfst_columns.extend(df.columns[12 : len(df.columns) - 1])
        df = df[reorder_pos_mfst_columns]
        return df
    
    def get_history_max_drawdown(self)->float:
        max_drawback = self.df_position_mfst['drawback_pct'].min()
        return max_drawback

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
        # df = self.df_choice_equd

        return mfst

    def get_choice_equ_by_date(self, trade_date: date) -> pd.DataFrame:
        df = self.df_choice_equd
        return df.loc[df.trade_date == trade_date, :]

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

    def get_choice_equ_metrics_by_list(
        self, choice: pd.DataFrame
    ) -> pd.DataFrame:
        df = self.df_choice_equd
        df.trade_date = pd.to_datetime(df.trade_date)
        # choice.ticker=choice.ticker.astype('string')
        choice.trade_date = pd.to_datetime(choice.trade_date)
        choice = pd.DataFrame.merge(
            choice, df, on=["ticker", "trade_date"], how="left"
        )
        return choice
