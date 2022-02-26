from typing import Callable
from kupy import configs

import pandas as pd
from sqlalchemy import Column, Float, Integer, Sequence, String
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base

Base: DeclarativeMeta = declarative_base()


class StrategyConfig:
    def __init__(self):
        self.data_folder=configs['data_folder'].data

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
        rebal_period,
        rebal_timing,
        mmf_enable,
        new_pos_target_pct,
        cash_pct,
        bench_num,
    ):
        # 调仓周期
        self.rebal_period: int = rebal_period
        # 调仓时间
        self.rebal_timing: int = rebal_timing  # 0 开盘， 1 收盘， 2 日均价
        self.mmf_enable: bool = mmf_enable
        self.new_pos_target_pct = new_pos_target_pct
        self.cash_pct = cash_pct
        # 备选股票数量
        self.bench_num = bench_num
        self.buy_criterial: Callable = None
        self.sale_criterial: Callable = None
        self.notsale_criterial: Callable = None

    def set_buy_criterial(self, shd_buy, *args):
        self.buy_criterial: Callable = shd_buy
        pass

    def set_sale_criterial(self, shd_sale, *args):
        self.sale_criterial: Callable = shd_sale
        pass

    def set_notsale_criterial(self, shdnot_sale, *args):
        self.notsale_criterial: Callable = shdnot_sale
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
