from datetime import date, datetime, timedelta
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys import *
import pandas as pd
from dys.domain import SelectMetric

from tests.my_test_strategy import MyETFStrategy, MyStrategy


given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


class TestBase:
    @pytest.fixture(scope="class")
    def db(self):
        return DBAdaptor()

    @pytest.fixture()
    def ms(self):
        return MyStrategy()

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_using_base_construct_ok(self, db: DBAdaptor):
        ms = MyStrategy()
        assert ms is not None
        df = ms.equd_pool
        equd_size = df.shape[0]
        logger.debug(f"current equ pool size:{ms.equ_pool.shape[0]}")
        logger.debug(f"current equd pool size:{equd_size}")

    def test_equ_pool_not_empty(self, db: DBAdaptor):
        mes: BaseStrategy = MyETFStrategy()
        assert mes.equ_pool is not None
        assert mes.equ_pool.shape[0] > 1
        logger.debug(f"current equ pool size:{mes.equ_pool.shape[0]}")
        logger.debug(f"current equd pool size:{mes.equd_pool.shape[0]}")

    def test_select_equd_by_date_without_customize_equ_pool(
        self, ms: MyStrategy
    ):
        ms.load_all_equ()
        ms.load_all_equd()
        df1 = ms._BaseStrategy__select_equd_by_date(date(2021, 1, 4))
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2021, 1, 4)
        df2 = ms._BaseStrategy__select_equd_by_date(
            date(2021, 1, 6), date(2022, 1, 28)
        )
        assert df2 is not None
        assert df2["trade_date"].iloc[0] == date(2021, 1, 6)
        assert df2["trade_date"].iloc[-1] == date(2022, 1, 28)

    def test_select_equd_by_date_with_customize_equ_pool(self, ms: MyStrategy):
        ms.set_equ_pool()
        df1: pd.DataFrame = ms._BaseStrategy__select_equd_by_date(
            date(2021, 1, 4)
        )
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2021, 1, 4)
        # 有退市的股票
        assert df1["ticker"].nunique() <= ms.df_equ_pool.shape[0]

    def test_select_equd_by_date_with_customize_equd_pool(
        self, ms: MyStrategy
    ):
        ms.set_equ_pool()
        ms.set_equd_pool()
        df1: pd.DataFrame = ms._BaseStrategy__select_equd_by_date(
            date(2021, 1, 4)
        )
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2021, 1, 4)
        # 有退市的股票
        assert df1["ticker"].nunique() <= ms.df_equ_pool.shape[0]

    def test__add_metric_column(self, ms: MyStrategy, db: DBAdaptor):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms._BaseStrategy__select_equd_by_date(date(2021,1,4))
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))

        ms._BaseStrategy__add_metric_column()
        df = ms.df_choice_equd
        assert "mom20" in df.columns
        logger.debug(df.columns)
        logger.debug(df)
        pass
    
    def test_select_equ_by_expression(self, ms:MyStrategy):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms._BaseStrategy__select_equd_by_date(date(2021,1,4))
        # ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms.set_select_equ_condition("close_price < 20")
        df = ms._BaseStrategy__select_equd_by_expression()
        assert df is not None
        logger.debug(df[['trade_date', 'ticker', 'close_price']])

    def test_ranking_1_factor(self, ms:MyStrategy):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms._BaseStrategy__select_equd_by_date(date(2022,1,4))
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))

        ms._BaseStrategy__add_metric_column()
        ms.append_rankfactor(RankFactor(name="mom20", bigfirst=True, weight=1))
        ms.post_hook_select_equ()
        df = ms.rank()
        assert "rank" in df.columns
        assert "mom20_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df.columns)
        logger.debug(df[['trade_date', 'ticker','mom20', 'rank', 'mom20_subrank']])        

    def test_ranking_1_factor_small_first(self, ms:MyStrategy):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms._BaseStrategy__select_equd_by_date(date(2022,1,4))

        ms.append_rankfactor(RankFactor(name="close_price", bigfirst = False, weight=1))
        ms.post_hook_select_equ()
        df = ms.rank()
        assert "rank" in df.columns
        assert "close_price_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df.columns)
        logger.debug(df[['trade_date', 'ticker','close_price', 'rank', 'close_price_subrank']])        

    def test_ranking_2_factor(self, ms:MyStrategy):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms.set_select_equ_condition("close_price<20")
        ms._BaseStrategy__select_equd_by_date(date(2022,1,4))
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms._BaseStrategy__add_metric_column()
        ms.post_hook_select_equ()
        ms._BaseStrategy__select_equd_by_expression()

        #由大到小排列
        ms.append_rankfactor(RankFactor(name="mom20", weight=2))
        #由小到大
        ms.append_rankfactor(RankFactor(name="close_price", bigfirst = False, weight=1))
        df = ms.rank()
        assert "rank" in df.columns
        assert "mom20_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df[['trade_date', 'ticker',   'close_price_subrank', 'mom20_subrank', 'rank']])        

    def test_generate_trade_mfst(self, my:MyStrategy):
        my
    def test_roi_mfst(self, ms:MyStrategy):

        ms.set_equ_pool()
        ms.set_equd_pool()
        ms.set_select_equ_condition("close_price<20")
        ms._BaseStrategy__select_equd_by_date(date(2022,1,4))
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms._BaseStrategy__add_metric_column()
        ms.post_hook_select_equ()
        ms._BaseStrategy__select_equd_by_expression()

        #由大到小排列
        ms.append_rankfactor(RankFactor(name="mom20", weight=2))
        #由小到大
        ms.append_rankfactor(RankFactor(name="close_price", bigfirst = False, weight=1))
        df = ms.rank()
        assert "rank" in df.columns
        assert "mom20_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df[['trade_date', 'ticker',   'close_price_subrank', 'mom20_subrank', 'rank']])        

