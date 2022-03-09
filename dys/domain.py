from typing import Callable

import pandas as pd
from kupy import configs
from sqlalchemy import Column, Float, Integer, Sequence, String
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base

Base: DeclarativeMeta = declarative_base()


class StrategyConfig:
    def __init__(self):
        self.data_folder = configs["data_folder"].data


class RankFactor:
    def __init__(
        self,
        name: str,
        bigfirst: bool = True,
        weight: int = 1,
    ):
        self.name: str = name
        self.bigfirst: bool = bigfirst  # True = asc False = desc
        self.weight: int = weight  # default value 1


class SelectMetric:
    def __init__(self, name: str, func: Callable, *args):
        if name == "":
            raise Exception("指标名称不能为空")
        self.name = name
        self.apply: Callable[[pd.DataFrame, tuple], pd.DataFrame] = func
        self.args = args


class TradeModel:
    def __init__(
        self,
        xperiod,
        xtiming,
        bench_num: int,
        unit_ideal_pos_pct: float,
        unit_pos_pct_tolerance: float,
        mini_unit_buy_pct: float,
        buy_fee_rate: float,
        sale_fee_rate: float,
        sale_rank: int = None,
        mmf_enable: bool = False,
    ):
        # 调仓周期
        self.xperiod: int = xperiod
        # 调仓时间 exchange timing
        self.xtiming: int = xtiming  # 0 开盘， 1 收盘， 2 日均价
        self.mmf_enable: bool = mmf_enable
        self.unit_ideal_pos_pct = unit_ideal_pos_pct * 0.01
        self.unit_pos_pct_tolerance = unit_pos_pct_tolerance * 0.01
        self.mini_unit_buy_pct = mini_unit_buy_pct * 0.01
        self.idea_unit_amount = (
            int(1 / unit_ideal_pos_pct) + 0
            if (1 / unit_ideal_pos_pct).is_integer()
            else 1
        )
        self.idea_max_amount = (
            int(1 / (unit_ideal_pos_pct + unit_pos_pct_tolerance)) + 0
            if (1 / (unit_ideal_pos_pct + unit_pos_pct_tolerance)).is_integer()
            else 1
        )
        # 买入手续费
        self.buy_fee_rate = buy_fee_rate
        # 卖出手续费
        self.sale_fee_rate = sale_fee_rate
        # 卖出排行
        self.sale_rank = sale_rank
        # 备选股票数量
        self.bench_num = bench_num
        self.buy_criterial = None
        self.sale_criterial = None
        self.notsale_criterial = None

    def append_buy_criterial(self, query_string):
        if self.buy_criterial is None:
            self.buy_criterial = query_string
        else:
            self.buy_criterial = self.buy_criterial + " and " + query_string

    def append_sale_criterial(self, query_string):
        if self.sale_criterial is None:
            self.sale_criterial = query_string
        else:
            self.sale_criterial = self.sale_criterial + " or " + query_string
        pass

    def append_notsale_criterial(self, query_string):
        if self.notsale_criterial is None:
            self.notsale_criterial = query_string
        else:
            self.notsale_criterial = (
                self.notsale_criterial + " or " + query_string
            )
        pass


class StrategyAccount(Base):
    __tablename__ = "strategy_account"
    __table_args__ = {"schema": "invest"}

    id = Column(
        Integer,
        Sequence("strategy_account_id_seq", schema="invest"),
        primary_key=True,
    )

    account_id = Column(Integer)
    account_name = Column(String(64))
    amount = Column(Float)
    trade_rate = Column(Float)
    pos_pct = Column(Float)

    def __init__(self):
        pass


class AccountData:
    def __init__(self):
        self.vol = 0
        self.amount = 0
        self.total_amount = 0
        self.balance = 0
        self.trade_fee = 0
        self.chg_pct = 0
        self.net = 0

    def __str__(self):
        return f"vol:{self.vol},amount:{self.amount},total_amount:{self.total_amount},balance:{self.balance},trade_fee:{self.trade_fee},chg_pct:{self.chg_pct},net:{self.net}"
