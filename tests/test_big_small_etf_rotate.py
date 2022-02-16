
from datetime import date, datetime, timedelta
from importlib import invalidate_caches
import math
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger
from dys.big_small_eft_rotate import BigSmallEtfRotateStrategy
import pandas as pd

from dys.domain import AccountData

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail

# 沪深300 510300 2012-05-28
# 易方达创业板ETF 159915 20111209
# _ticker_x = "'510300'"  # 沪深300
# _ticker_x = "'510050'" #
# _ticker_y = "'159915'" # 创业板指数
class TestBigSmallEtfRotateStrategy:
    @pytest.fixture(scope="class")
    def db(self):
        return DBAdaptor(is_use_cache=True)
    
    @pytest.fixture(scope="class")
    def bsery(self):
        return BigSmallEtfRotateStrategy(1)

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        db = DBAdaptor()
        db.execute_any_sql("delete from invest.big_small_etf_rotate")
        yield
        logger.info("TestCase Level Tear Down is triggered!")
        
    def test_strategy_construct_ok(self, db:DBAdaptor):
        bsery = BigSmallEtfRotateStrategy(1) 
        assert bsery is not None
        assert bsery.account.account_id == 1
        assert bsery.ticker_x != ""
        assert bsery.ticker_y != ""
        assert bsery.trade_rate > 1/10000
        assert bsery.initial_amount >= 10000
        assert bsery.momentum_days > 1
        assert bsery.ticker_x_name != ""
        assert bsery.ticker_y_name != ""
        logger.debug(bsery)

    
    def test_get_strategy_by_latest_date(self, bsery:BigSmallEtfRotateStrategy):
        """指定日期2022/1/28 (收盘后运行)，算出轮动风格以及第二日的买卖策略

        Args:
            bsery (BigSmallEtfRotateStrategy): _description_
        """        
        start_date=date(2022,1,28)
        end_date=start_date
        expect = bsery.get_strategy_by_date(start_date, end_date)
        assert expect is not None
        assert expect.shape[0] == 1
        assert expect.index[0] == start_date
        assert expect["style"].iloc[0] == "empty"
        assert expect["pos"].iloc[0] == "empty"
        assert expect["strategy_chg_pct_adj"].iloc[0] == 0.0
        assert expect["strategy_net"].iloc[0] == 1.0
        assert expect["trade_cd"].iloc[0] == 0
        expect.to_csv("/tmp/bsery_can_run.csv")
        logger.info(f"{expect}")

    def test_get_strategy_by_nonlatestdate(self, bsery:BigSmallEtfRotateStrategy):
        """指定日期2022/1/27 (收盘后运行)，算出轮动风格以及第二日的买卖策略

        Args:
            bsery (BigSmallEtfRotateStrategy): _description_
        """        
        start_date=date(2020,1,20)
        end_date=date(2022,1,28)
        expect = bsery.get_strategy_by_date(start_date, end_date)
        expect.to_csv("/tmp/bsery_can_run.csv")
        assert expect is not None
        assert expect.shape[0] > 1
        assert expect.index[0] == start_date
        assert expect["pos"].iloc[0] == "empty"
        assert expect["pos"].iloc[1] == "small"
        # assert expect["strategy_chg_pct_adj"].iloc[1] == 0.0
        # assert expect["strategy_net"].iloc[1] == 1.0
        # assert expect["trade_cd"].iloc[0] == 0
        logger.info(f"{expect}")

    def test_write_df_to_db_latest_row(self,  bsery:BigSmallEtfRotateStrategy):
        start_date=date(2022,1,28)
        end_date=date(2022,1,28)
        expect = bsery.get_strategy_by_date(start_date, end_date)
        bsery.write_df_to_db(expect)
        db = DBAdaptor()
        actual = db.get_df_by_sql("select * from invest.big_small_etf_rotate")
        assert expect is not None
        assert actual is not None

        assert expect.shape[0] == actual.shape[0]

        assert actual["trade_date"].iloc[0] != ""
        assert actual["pos"].iloc[0] == "empty"
        assert actual["style"].iloc[0] == "empty"
        assert actual["strategy_chg_pct_real"].iloc[0] == 0.0
        assert actual["strategy_net_real"].iloc[0] == 1.0
        assert actual["total_amount"].iloc[0] == bsery.account.amount
        assert actual["amount"].iloc[0] == 0.0
        assert actual["vol"].iloc[0] == 0

        
    def test_calc_df_with_account_data(self, bsery:BigSmallEtfRotateStrategy):

        start_date=date(2020,1,20)
        end_date=date(2022,1,28)
        df = bsery.get_strategy_by_date(start_date, end_date)
        # df.to_pickle("/tmp/test_calc_df_with_account.pkl")
        # df:pd.DataFrame = pd.read_pickle("/tmp/test_calc_df_with_account.pkl")
        df.reset_index(inplace=True)

        initdata:AccountData = AccountData()
        initdata.total_amount=100000
        initdata.balance=100000
        initdata.net=1
    
        df = bsery.calc_df_with_account_data(initdata, df)

        
        df.to_csv("/tmp/test_calc_df_with_account.csv")

        
    def test_write_df_to_db_many_row(self,  bsery:BigSmallEtfRotateStrategy):
        start_date=date(2020,1,20)
        end_date=date(2022,1,28)
        expect = bsery.get_strategy_by_date(start_date, end_date)
        bsery.write_df_to_db(expect)
        db = DBAdaptor()
        actual = db.get_df_by_sql("select * from invest.big_small_etf_rotate order by id")
        assert expect is not None
        assert actual is not None

        assert actual.shape[0] >1

        assert actual["trade_date"].iloc[0] == start_date
        assert actual["pos"].iloc[0] == "empty"
        assert actual["pos"].iloc[1] == "small"
        actual.to_csv("/tmp/db_export.csv")
    
    def test_run_two_time(self,bsery:BigSmallEtfRotateStrategy):
        start_date=date(2022,1,24)
        end_date = date(2022,1,26)
        expect1 = bsery.get_strategy_by_date(start_date, end_date)
        bsery.write_df_to_db(expect1)
        start_date=date(2022,1,27)
        end_date = date(2022,1,28)
        expect2 = bsery.get_strategy_by_date(start_date, end_date)
        bsery.write_df_to_db(expect2)
        db = DBAdaptor()
        actual = db.get_df_by_sql("select * from invest.big_small_etf_rotate order by id")
        assert expect1 is not None
        assert expect2 is not None
        assert actual is not None

        assert expect1.shape[0] == 3

        assert expect2.shape[0] == 2

        assert actual.shape[0] == 5


    # def test_stragegy_can_run_as_daemon_but_suspend(self, db:DBAdaptor):
    #     #假设30天前开始轮动
    #     bsery1 = BigSmallEtfRotateStrategy(1)
    #     start_date=datetime.today().date() -timedelta(30)
    #     bsery1.run_as_daemon(start_date)
    #     #运行20天，然后suspend， 10天
    #     bsery2 = BigSmallEtfRotateStrategy(1)
    #     bsery2.set_suspend_status(True)
    #     start_date=datetime.today().date() -timedelta(10)
    #     bsery2.run_as_daemon(start_date)
    #     #suspend 5天之后，再恢复
    #     bsery3 = BigSmallEtfRotateStrategy(1)
    #     bsery3.set_suspend_status(False)
    #     start_date=datetime.today().date() -timedelta(5)
    #     bsery3.run_as_daemon(start_date)

    # # @skip
    # def test_stragegy_can_run(self, db:DBAdaptor):
    #     bsery = BigSmallEtfRotateStrategy(1)
    #     bsery.run_with_plot(datetime.today().date())
