from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Sequence,
    String,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BigSmallEtfRotate(Base):
    __tablename__ = "big_small_etf_rotate"
    __table_args__ = {"schema": "invest"}

    id = Column(
        Integer,
        Sequence("big_small_etf_rotate_id_seq", schema="invest"),
        primary_key=True,
    )

    trade_date = Column(DateTime)
    account_id = Column(Integer)
    account_name = Column(Integer)
    strategy_status_cd = Column(String(2)) #策略是否暂停实施
    ticker_x = Column(String(16))
    sec_short_name_x = Column(String(64))
    pre_close_price_x = Column(Float)
    open_price_x = Column(Float)
    close_price_x = Column(Float)
    # chg_x = Column(Float)
    chg_pct_x = Column(Float)
    ticker_y = Column(String(16))
    sec_short_name_y = Column(String(64))
    pre_close_price_y = Column(Float)
    open_price_y = Column(Float)
    close_price_y = Column(Float)
    # chg_y = Column(Float)
    chg_pct_y = Column(Float)
    big_mom = Column(Float)
    small_mom = Column(Float)
    style = Column(String(64)) #风格 
    pos = Column(String(64)) #持仓
    strategy_chg_pct = Column(Float) # 扣除手续费后的涨幅
    trade_cd = Column(Integer) # 扣除手续费后的涨幅
    next_trade_cd = Column(Integer) # 扣除手续费后的涨幅
    big_net = Column(Float) # 净值因子
    small_net = Column(Float) # 净值因子
    strategy_net = Column(Float) # 净值因子
    trade_fee = Column(Float) # 扣除手续费后的涨幅
    vol = Column(Float) # 交易多少手
    amount = Column(Float) # 资产市值
    total_amount = Column(Float) # 资产市值
    strategy_chg_pct_real = Column(Float) # 扣除手续费后的涨幅
    strategy_net_real = Column(Float) # 净值因子
    update_time = Column(DateTime)  #     更新时间
    comment = Column(String(255))  # 更新说明

    def __init__(self):
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
