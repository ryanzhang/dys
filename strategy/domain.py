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


class SyncStatus(Base):
    __tablename__ = "sync_status"
    __table_args__ = {"schema": "stock"}

    id = Column(
        Integer,
        Sequence("sync_status_id_seq", schema="stock"),
        primary_key=True,
    )
    rc = Column(Boolean)
    table_name = Column(String(64))  #     证券交易所
    update_time = Column(DateTime)  #     日期
    comment = Column(String(255))  # 更新说明

    def __init__(self):
        pass

class BigSmallEtfRotate(Base):
    __tablename__ = "big_small_etf_rotate"
    __table_args__ = {"schema": "invest"}

    id = Column(
        Integer,
        Sequence("bser_id_seq", schema="invest"),
        primary_key=True,
    )

    trade_date = Column(DateTime)
    account_id = Column(Integer)
    is_suspend = Column(Integer) #策略是否暂停实施
    is_skip_trade = Column(Integer) #是否跳过量化交易
    ticker_x = Column(String(16))
    pre_close_price_x = Column(Float)
    open_price_x = Column(Float)
    close_price_x = Column(Float)
    chg_x = Column(Float)
    chg_pct_x = Column(Float)
    ticker_y = Column(String(16))
    pre_close_price_y = Column(Float)
    open_price_y = Column(Float)
    close_price_y = Column(Float)
    chg_x = Column(Float)
    chg_pct_y = Column(Float)
    big_mom = Column(Float)
    small_mom = Column(Float)
    style = Column(String(64)) #风格 
    pos = Column(String(64)) #持仓
    pos_chg_pct = Column(Float) # 扣除手续费后的涨幅
    trade_fee = Column(Float) # 扣除手续费后的涨幅
    big_net = Column(Float) # 净值因子
    small_net = Column(Float) # 净值因子
    strategy_net = Column(Float) # 净值因子
    amount = Column(Float) # 资产市值
    update_time = Column(DateTime)  #     更新时间
    comment = Column(String(255))  # 更新说明

    def __init__(self):
        pass
