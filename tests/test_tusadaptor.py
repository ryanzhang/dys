from datetime import date, datetime, timedelta
from syd.dbadaptor import DBAdaptor
import pytest
from syd.tusadaptor import TUSAdaptor
import os

from syd.logger import logger
from syd.config import configs

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail

expect_cache_filepath = configs["cache_folder"].data + "tus_stock_basic.pkl"


class TestTUSAdaptor:
    @pytest.fixture(scope="class")
    def tus(self):
        logger.info("Setup for Class")
        tus = TUSAdaptor(is_use_cache=True, is_export_csv=True)
        return tus

    def test_get_stock_basic_with_cache(self, tus):
        df, csv_filepath = tus.getStockBasicInfo()
        assert df is not None and df.size > 0, "返回的df不能为空"
        assert csv_filepath is not None
        assert expect_cache_filepath is not None
        assert os.path.exists(expect_cache_filepath)
        assert os.path.exists(csv_filepath)

    def test_ts_code_to_sec_id(self):
        assert "000001.XSHE" == TUSAdaptor.ts_code_to_sec_id("000001.SZ")
        assert "600000.XSHG" == TUSAdaptor.ts_code_to_sec_id("600000.SH")

    # @skip
    def test_get_mkt_equ_daily(self, tus: TUSAdaptor):
        trade_date = date(2022, 1, 28)
        df = tus.getMktEquD(trade_date)
        assert df is not None and df.shape[0] > 0, "返回的df不能为空"
        df.to_csv("/tmp/df_20220128.csv")

    def test_get_mkt_equ_d_by_codelist(self, tus: TUSAdaptor):
        trade_date = date(2022, 1, 28)
        db = DBAdaptor()
        df_expect = db.getDfBySql(
            "select id, sec_id, list_date from stock.equity where list_status_cd='L' and exchange_cd='XSHG' order by id desc limit 2"
        )
        start_date = df_expect["list_date"].min()
        end_date = df_expect["list_date"].max()
        assert start_date < datetime.today().date()
        assert end_date < datetime.today().date()
        assert start_date < end_date
        df = tus.getMktEquDByCodeList(
            df_expect["sec_id"], start_date=start_date, end_date=end_date
        )
        assert df is not None and df.shape[0] > 0, "返回的df不能为空"
        df.to_csv("/tmp/df_incr_daterange.csv")

    # @skip
    # def test_get_stock_basic_without_cache(self,tus):
    #     #默认是没有cache的
    #     tus.setCacheMode(False)
    #     df,csv_filepath = tus.getStockBasicInfo()
    #     assert df is not None
    #     assert csv_filepath is None


if __name__ == "__main__":
    pytest.main(args=["-s", os.path.abspath(__file__)])
