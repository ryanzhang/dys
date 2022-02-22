from datetime import date, datetime, timedelta
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
        df = db.get_df_by_sql("select * from stock.mkt_equ_day where trade_date = '20220104'")
        assert df is not None
        return df


    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        db= DBAdaptor(is_use_cache=True)
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_momentum(self, df:pd.DataFrame):
        sm = SelectMetric("mom20", m.momentum, 20, "close_price")
        df[sm.name] = sm.apply(df, sm.args)
        assert "mom20" in df.columns
        logger.debug(df)
        pass
    
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
