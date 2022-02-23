from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from importlib import invalidate_caches
import math
import pytest
import pandas as pd
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.bs import BaseStrategy
from dys.domain import RankFactor, SelectMetric
from dys.metric import m


given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


class TestMetrics:
    @pytest.fixture()
    def df(self):
        db = DBAdaptor(is_use_cache=True)
        df = db.get_df_by_sql("select * from stock.mkt_equ_day where trade_date > '20210104'")
        assert df is not None
        return df


    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        # db= DBAdaptor(is_use_cache=True)
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_momentum(self, df:pd.DataFrame):
        sm = SelectMetric("mom20", m.momentum, 20, "close_price")
        df[sm.name] = sm.apply(df, sm.args)
        assert sm.name in df.columns
        logger.debug(df)
        
    
    def test_list_days(self, df:pd.DataFrame):
        db = DBAdaptor()
        df_list_days=db.get_df_by_sql("select ticker, list_date from stock.equity where list_status_cd ='L'")
        sm = SelectMetric("list_days", m.list_days, df_list_days )
        df[sm.name] = sm.apply(df, sm.args)
        assert "list_days" in df.columns
        logger.debug(df)

    def test_wq_alpha16(self, df:pd.DataFrame):
        # db = DBAdaptor()
        sm = SelectMetric("wq_alpha16", m.wq_alpha16)
        df[sm.name] = sm.apply(df, sm.args)
        assert "wq_alpha16" in df.columns
        logger.debug(df)

    def test_ntra_turnover_rate(self, df: pd.DataFrame):
        sm = SelectMetric("ntra_turnover_rate", m.ntra_turnover_rate)
        df[sm.name] = sm.apply(df, sm.args)
        assert "ntra_turnover_rate" in df.columns
        assert not df[sm.name].isna()
        logger.debug(df)
        

    def test_bias(self, df: pd.DataFrame):
        sm = SelectMetric("bias", m.bias, 6)
        df[sm.name] = sm.apply(df, sm.args)
        assert "bias" in df.columns

        df_000001=df[df.ticker=='000001']
        df_000001.set_index('trade_date',inplace=True)
        df_000001.dropna(inplace=True)
        logger.debug(df_000001[['close_price', 'bias']])
        # df_000001.to_csv("/tmp/test_bias_000001.csv")

        # plt.plot(df_000001["bias"], label='bias')
        # plt.plot(df_000001["close_price"], label='close_price')
        # plt.show()
        

    def test_float_rate_n(self, df: pd.DataFrame ):
        db = DBAdaptor(is_use_cache=True)
        N=90
        df_float = db.get_df_by_sql("select * from stock.equ_share_float")
        sm = SelectMetric(f"float_rate_{N}", m.float_rate, N, df_float)
        df[sm.name] = sm.apply(df, sm.args)

        # df.to_csv("/tmp/test_float_rate_all.csv")

        df_000156=df[df.ticker=='000156']
        df_000156.set_index('trade_date',inplace=True)
        df_000156.dropna(inplace=True)
        logger.debug(df_000156[['ticker',f'float_rate_{N}']])
        # df_000156.to_csv("/tmp/test_float_rate_000156.csv")

        


    def test_float_value_n(self, df: pd.DataFrame):
        db = DBAdaptor(is_use_cache=True)
        N=90
        df_float = db.get_df_by_sql("select * from stock.equ_share_float")
        sm = SelectMetric('float_rate_{N}', m.float_value, N, df_float)
        df[sm.name] = sm.apply(df, sm.args)

        df_000156=df[df.ticker=='000156']
        df_000156.set_index('trade_date',inplace=True)
        df_000156.dropna(inplace=True)
        logger.debug(df_000156[['ticker','float_rate_'+str(N)]])
        # df_000156.to_csv("/tmp/test_bias_000156.csv")
        


    def test_chg_pct_sum(self, df: pd.DataFrame):
        N=20
        sm = SelectMetric(f'SUM{N}_chg_pct', m.chg_pct_sum, N)
        df[sm.name] = sm.apply(df, sm.args)
        assert sm.name in df.columns

        





























