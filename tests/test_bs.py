from datetime import date, datetime, timedelta
import os
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


class TestBaseStrategy:
    # 加载一年的数据，并加载120天的margin 用于计算指标
    @pytest.fixture(scope="class")
    def ms(self):
        ms = MyStrategy('20200709','20211231') 
        # Modify the config.data_folder
        ms._BaseStrategy__set_cache_folder(ms.config.data_folder + "tests/")
        return ms

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_using_default_construct_ok(self):
        ms = MyStrategy()
        assert ms is not None
        assert ms.df_equ_pool is not None
        assert ms.df_equd_pool is not None
        assert ms.df_choice_equd is not None
        assert ms.config is not None
        assert ms.config.data_folder is not None
        assert os.path.exists(ms.config.data_folder+"/cache/6f258.parquet")

    def test_customize_date_construck_ok(self):
        start_date='20210104'
        end_date='20211229'
        logger.debug(f"Construct customize date from {start_date} to {end_date}")
        ms = MyStrategy(start_date, end_date) 
        assert ms is not None
        assert ms.df_equ_pool is not None
        assert ms.df_equd_pool is not None
        assert ms.df_choice_equd is not None

        logger.debug(f"current equ pool size:{ms.df_equ_pool.shape[0]}")
        logger.debug(f"current equd pool size:{ms.df_equd_pool.shape[0]}")
    

    def test_select_equd_by_date_without_customize_equ_pool(
        self, ms: MyStrategy
    ):
        
        df1 = ms.select_equd_by_date(date(2020, 7, 9))
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2020, 7, 9)
        df2 = ms.select_equd_by_date(
            date(2021, 7, 6), date(2021, 8, 6)
        )
        assert df2 is not None
        assert df2["trade_date"].iloc[0] == date(2021, 7, 6)
        assert df2["trade_date"].iloc[-1] == date(2021, 8, 6)

    def test_select_equd_by_date_with_customize_equ_pool(self, ms: MyStrategy):
        # set_equ_pool custimze the equ poo to only XSHG
        ms.set_equ_pool()
        df1: pd.DataFrame = ms.select_equd_by_date(
            date(2021, 7, 9)
        )
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2021, 7, 9)
        # 有退市的股票
        n = df1["ticker"].nunique() 
        assert n <= ms.df_equ_pool.shape[0]
        assert df1.loc[df1.sec_id.str.endswith('XSHG'),:].shape[0] == df1.shape[0]
        assert df1["ticker"].nunique() <= ms.df_equ_pool.shape[0]

    def test__add_metric_column(self, ms: MyStrategy):
        ms.append_metric(SelectMetric("MOM20_close_price", m.momentum, 20, "close_price"))
        df = ms.df_choice_equd
        assert "MOM20_close_price" in df.columns
        assert os.path.exists(ms.config.data_folder + "metrics/MOM20_close_price.paquet")
        logger.debug(df)

    def test_select_equ_by_expression(self, ms: MyStrategy):
        ms.select_equd_by_date(date(2021, 1, 4))
        # ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms.append_select_condition("close_price < 20")
        df = ms.df_choice_equd
        assert df is not None
        rowcount = df.shape[0]
        assert rowcount == df.loc[df['close_price']<20,:].shape[0]

        logger.debug(df[["trade_date", "ticker", "close_price"]])

    def test_select_by_date(self, ms: MyStrategy):
        ms.select_equd_by_date(date(2021, 3, 4))
        df = ms.df_choice_equd

        assert df.iloc[0].trade_date == date(2021,3,4)
        logger.debug(df)

    def test_ranking_1_factor(self, ms: MyStrategy):
        ms.append_metric(SelectMetric("MOM20_close_price", m.momentum, 20, "close_price"))

        ms.select_equd_by_date(date(2022, 1, 5))

        ms.append_rankfactor(RankFactor(name="MOM20_close_price", bigfirst=True, weight=1))
        ms.post_hook_select_equ()
        df = ms.rank()
        assert "rank" in df.columns
        assert "MOM20_close_price_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df.columns)
        logger.debug(
            df[["trade_date", "ticker", "MOM20_close_price", "rank", "MOM20_close_price_subrank"]]
        )

    def test_ranking_1_factor_small_first(self, ms: MyStrategy):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms.select_equd_by_date(date(2022, 1, 4))

        ms.append_rankfactor(
            RankFactor(name="close_price", bigfirst=False, weight=1)
        )
        ms.post_hook_select_equ()
        df = ms.rank()
        assert "rank" in df.columns
        assert "close_price_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df.columns)
        logger.debug(
            df[
                [
                    "trade_date",
                    "ticker",
                    "close_price",
                    "rank",
                    "close_price_subrank",
                ]
            ]
        )

    def test_ranking_2_factor(self, ms: MyStrategy):
        ms.set_equ_pool()
        ms.set_equd_pool()
        ms.set_select_equ_condition("close_price<20")
        ms.select_equd_by_date(date(2022, 1, 4))
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms._BaseStrategy__add_metric_column()
        ms.post_hook_select_equ()
        ms._BaseStrategy__select_equd_by_expression()

        # 由大到小排列
        ms.append_rankfactor(RankFactor(name="mom20", weight=2))
        # 由小到大
        ms.append_rankfactor(
            RankFactor(name="close_price", bigfirst=False, weight=1)
        )
        df = ms.rank()
        assert "rank" in df.columns
        assert "mom20_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(
            df[
                [
                    "trade_date",
                    "ticker",
                    "close_price_subrank",
                    "mom20_subrank",
                    "rank",
                ]
            ]
        )

    def test_choice_equ_by_date(self, ms: MyStrategy):
        the_date = date(2021,1,5)
        df = ms.get_choice_equ_by_date(the_date)
        assert df is not None
        logger.debug(f'{the_date} 共选出 {df.shape[0]}')

    def test_generate_trade_mfst(self, my: MyStrategy):
        mfst = my.generate_trade_mfst()
        assert mfst is not None

    def test_get_trade_mfst_by_date(self, my: MyStrategy):
        the_date = date(2021,1,5)
        df = my.get_trade_mfst_by_date(the_date)
        assert df is not None
        logger.debug(f'{the_date} 共选出 {df.shape[0]}')

    def test_roi_mfst(self, ms: MyStrategy):

        ms.set_equ_pool()
        ms.set_equd_pool()
        ms.set_select_equ_condition("close_price<20")
        ms.select_equd_by_date(date(2022, 1, 4))
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms._BaseStrategy__add_metric_column()
        ms.post_hook_select_equ()
        ms._BaseStrategy__select_equd_by_expression()

        # 由大到小排列
        ms.append_rankfactor(RankFactor(name="mom20", weight=2))
        # 由小到大
        ms.append_rankfactor(
            RankFactor(name="close_price", bigfirst=False, weight=1)
        )
        df = ms.rank()
        assert "rank" in df.columns
        assert "mom20_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(
            df[
                [
                    "trade_date",
                    "ticker",
                    "close_price_subrank",
                    "mom20_subrank",
                    "rank",
                ]
            ]
        )
