# -*- coding: UTF-8 -*-
from syd.domain import MktIdxDay
import pytest
from syd.stocksyncer import *
from datetime import datetime, date
from syd.logger import logger

# 测试equity表的更细情况

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


def test_tus_code_split():
    assert TUSAdaptor.tus_code_split("000001.SZ")[0] == "000001"
    assert TUSAdaptor.tus_code_split("000001.SZ")[1] == "XSHE"

    assert TUSAdaptor.tus_code_split("000001.SH")[0] == "000001"
    assert TUSAdaptor.tus_code_split("000001.SH")[1] == "XSHG"

    assert TUSAdaptor.tus_code_split("000001.BJ")[0] == "000001"
    assert TUSAdaptor.tus_code_split("000001.BJ")[1] == "XBEI"


def test_sector_cd_lambda():
    assert 1 == (lambda x: {"主板": 1, "创业版": 2, "科创版": 4, "北交所": 5}[x])("主板")
    assert 5 == (lambda x: {"主板": 1, "创业版": 2, "科创版": 4, "北交所": 5}[x])("北交所")


class TestStockSync:
    @pytest.fixture(scope="class")
    def syncer(self):
        logger.info("Setup for Class")
        syncer = StockSyncer(is_export_csv=True)
        db = DBAdaptor()
        return syncer

    @pytest.fixture(scope="class")
    def db(self):
        db = DBAdaptor()
        return db

    def test_equity_has_been_sync_to_latest(
        self, syncer: StockSyncer, db: DBAdaptor
    ):
        df_incremental, export_csv_list = syncer.sync_equity()

        logger.info("exported csv:" + str(export_csv_list))

        df_tus = pd.read_csv(export_csv_list[0])
        df_db_bf = pd.read_csv(export_csv_list[1])

        assert df_incremental is not None, "update 不能返回None"
        df_incremental.to_csv(f"/tmp/df_incremental.csv")

        df_db_latest = db.getDfBySql("select ticker from stock.equity")
        assert df_db_latest["ticker"].is_unique, "不能有重复的股票"

        sizediff = df_db_latest.shape[0] - df_db_bf.shape[0]

        assert sizediff == df_incremental.shape[0], "增量更新应该等于原来的差值"
        df_db_sync = db.getDfBySql(
            "select * from stock.sync_status where table_name= 'equity'"
        )
        assert df_db_sync is not None
        assert (
            df_db_sync.iloc[0]["update_time"].date() == date.today()
        ), "sync_status已经被更新"

    def test_trade_calendar_has_been_sync_to_latest(
        self, syncer: StockSyncer, db: DBAdaptor
    ):
        df_incremental = syncer.sync_trade_calendar()
        assert df_incremental is not None, "update 不能返回None"

        df_db = db.getDfBySql(
            "select calendar_date from stock.trade_calendar order by trade_calendar desc limit 1"
        )
        assert df_db is not None

        if df_incremental.shape[0] != 0:
            assert (
                df_db.iloc[0]["calendar_date"] == df_incremental.index[-1]
            ), "最新的日期已经同步到数据库中"
        else:
            logger.info("trade_calendar已经更新到今年最后一天了")

        df_db_sync = db.getDfBySql(
            "select * from stock.sync_status where table_name= 'trade_calendar'"
        )
        assert df_db_sync is not None
        assert (
            df_db_sync.iloc[0]["update_time"].date() == date.today()
        ), "sync_status已经被更新"
        assert df_db_sync.iloc[0]["comment"] != "", "sync_status comment 字段不为空"

    def test_get_latest_trade_date(self, syncer: StockSyncer):
        ret = syncer.getLatestTradeDate()
        logger.info("最后一个交易日:" + str(ret))
        assert ret <= datetime.today().date()

    def test_is_a_date_open_or_not(self, syncer: StockSyncer):
        assert syncer.is_open_day(date(2022, 1, 28))
        assert not syncer.is_open_day(date(2022, 1, 29))
        assert syncer.is_open_day(date(2022, 1, 27))

    def test_mkt_equ_day_basic_has_been_fetch_from_tus(
        self, syncer: StockSyncer
    ):
        df = syncer.sync_mkt_equity_day()
        df.to_pickle("/tmp/mkt_equ_day_15_28.pkl")
        df.to_csv("/tmp/mkt_equ_day_15_28.csv")
        assert df is not None and df.shape[0] > 0

    def test_mkt_equ_day_extra_has_been_fetch_from_tus(
        self, syncer: StockSyncer
    ):
        df = syncer.sync_mkt_equity_day()
        df.to_pickle("/tmp/mkt_equ_day_extra_15_28.pkl")
        df.to_csv("/tmp/mkt_equ_day_15_28.csv")
        assert df is not None and df.shape[0] > 0

    # 复权
    def test_mkt_equ_day_hfq_has_been_fetch_from_tus(
        self, syncer: StockSyncer
    ):
        df = syncer.sync_mkt_equity_day()
        df.to_pickle("/tmp/mkt_equ_day_hfq_15_28.pkl")
        df.to_csv("/tmp/mkt_equ_day_hfq_15_28.csv")
        assert df is not None and df.shape[0] > 0

    def test_new_equ_day_has_been_fetch_from_tus(self, syncer: StockSyncer):
        df = syncer.get_new_equ_mktday()
        df.to_pickle("/tmp/mkt_equ_day_new.pkl")
        df.to_csv("/tmp/mkt_equ_day_new.csv")
        assert df is not None and df.shape[0] > 0

    def test_merge_tus_df(self, db: DBAdaptor):
        df1 = pd.read_pickle("/tmp/mkt_equ_day_15_28.pkl")
        df2 = pd.read_pickle("/tmp/mkt_equ_day_extra_15_28.pkl")
        df2 = df2[["ts_code", "trade_date", "pe", "pb"]]
        df3 = pd.read_pickle("/tmp/mkt_equ_day_hfq_15_28.pkl")
        df3 = df3[["ts_code", "trade_date", "adj_factor"]]
        df = pd.merge(df1, df2, how="left", on=["ts_code", "trade_date"])
        df = pd.merge(df, df3, how="left", on=["ts_code", "trade_date"])

        # 比较datayes与tus后复权差异
        # 有36只股票后复权存在差异，目前先忽略这种差异，保留结果以后再查
        # df_dy_hfq = db.getDfBySql("select sec_id, trade_date, accum_adj_af_factor from stock.mkt_equ_day where trade_date='20220114'")
        # df_tus_hfq = df3.loc[df3.trade_date == '20220117',:]
        # df_tus_hfq.rename(columns={'ts_code':'sec_id'}, inplace=True)
        # df_tus_hfq['sec_id'] = df_tus_hfq['sec_id'].apply(lambda x : TUSAdaptor.ts_code_to_sec_id(x))
        # df_hfq_merge=pd.merge(df_dy_hfq, df_tus_hfq, how='left', on=['sec_id'])
        # df_hfq_merge['hfq_diff'] = df_hfq_merge['accum_adj_af_factor'] - df_hfq_merge['adj_factor']
        # df_hfq_diff=df_hfq_merge[df_hfq_merge['hfq_diff']>1.0]
        # df_hfq_merge.to_csv("/tmp/df_hfq_merge.csv")
        # df_hfq_diff.to_csv("/tmp/df_hfq_diff.csv")

        # assert df is not None and df.shape[0]>0
        # df.sort_values([ 'trade_date', 'ts_code'], inplace=True)
        # df.to_csv("/tmp/df_merged_tus_15_28.csv")

    def test_mkt_equ_day_has_been_update_to_latest(
        self, syncer: StockSyncer, db: DBAdaptor
    ):
        df1 = pd.read_pickle("/tmp/mkt_equ_day_15_28.pkl")
        df2 = pd.read_pickle("/tmp/mkt_equ_day_extra_15_28.pkl")
        df2 = df2[["ts_code", "trade_date", "pe", "pb"]]
        df3 = pd.read_pickle("/tmp/mkt_equ_day_hfq_15_28.pkl")
        df3 = df3[["ts_code", "trade_date", "adj_factor"]]
        df = pd.merge(df1, df2, how="left", on=["ts_code", "trade_date"])
        df = pd.merge(df, df3, how="left", on=["ts_code", "trade_date"])

        # syncer.write_to_db(df)
        df_result = db.getDfBySql(
            "select * from stock.mkt_equ_day \
            where ticker='000001' order by trade_date desc limit 30"
        )
        assert df_result is not None
        assert df_result.shape[0] > 1
        assert (
            df_result.iloc[0]["trade_date"] == syncer.getLatestTradeDate()
        ), "数据库中没有找到最后一个交易日数据"
        assert df_result.iloc[0]["sec_id"] is not None, "字段不能空"
        assert df_result.iloc[0]["ticker"] is not None, "字段不能空"
        assert df_result.iloc[0]["sec_short_name"] is not None, "字段不能空"
        assert df_result.iloc[0]["exchange_cd"] is not None, "字段不能空"
        assert df_result.iloc[0]["trade_date"] is not None, "字段不能空"
        assert df_result.iloc[0]["pre_close_price"] is not None, "字段不能空"
        assert df_result.iloc[0]["open_price"] is not None, "字段不能空"
        assert df_result.iloc[0]["highest_price"] is not None, "字段不能空"
        assert df_result.iloc[0]["lowest_price"] is not None, "字段不能空"
        assert df_result.iloc[0]["close_price"] is not None, "字段不能空"
        assert df_result.iloc[0]["turnover_vol"] is not None, "字段不能空"
        assert df_result.iloc[0]["turnover_value"] is not None, "字段不能空"
        assert df_result.iloc[0]["turnover_rate"] is not None, "字段不能空"
        assert df_result.iloc[0]["neg_market_value"] is not None, "字段不能空"
        assert df_result.iloc[0]["market_value"] is not None, "字段不能空"
        assert df_result.iloc[0]["chg_pct"] is not None, "字段不能空"
        assert df_result.iloc[0]["pe"] is not None, "字段不能空"
        assert df_result.iloc[0]["pe1"] is not None, "字段不能空"
        assert df_result.iloc[0]["pb"] is not None, "字段不能空"
        assert df_result.iloc[0]["is_open"] is not None, "字段不能空"
        assert df_result.iloc[0]["vwap"] is not None, "字段不能空"
        # assert df_result.iloc[0]['accum_adj_bf_factor'] is not None , "字段不能空"
        # assert df_result.iloc[0]['accum_adj_af_factor'] is not None , "字段不能空"

        df_sync_status = db.getDfBySql(
            "select * from stock.sync_status \
            where tablename='mkt_equ_day'"
        )
        assert df_sync_status is not None
        assert df_sync_status.iloc[0]["rc"] == True, "return code字段应该为True"
        assert df_sync_status.iloc[0]["comment"] is not None, "字段不能空"
        assert (
            df_sync_status.iloc[0]["update_time"].date()
            == datetime.today().date()
        ), "字段不能空"

    @skip
    def test_mkt_equity_day_integrity(
        self, db: DBAdaptor, syncer: StockSyncer
    ):
        pass
        # df = syncer.get_missing_equ_day_data()
        # df.to_pickle("/tmp/missing_mkt_equ_d.pkl")
        # df.to_csv("/tmp/missing_mkt_equ_d.csv")
        # df = pd.read_pickle("/tmp/missing_mkt_equ_d.pkl")
        # syncer.write_to_db(df)
        # df_code_expect = db.getDfBySql("select sec_id, list_date from stock.equity where list_status_cd='L' order by sec_id")
        # df_code_actual = db.getDfBySql("select distinct sec_id from stock.mkt_equ_day order by sec_id")
        # df_incr = df_code_expect[~df_code_expect.sec_id.isin(df_code_actual.sec_id)]
        # df_incr.sort_values(['list_date', 'sec_id'], inplace=True)
        # df_incr.to_csv("/tmp/df_incremental.csv")

    def test_sync_mkt_equ_d(self, syncer: StockSyncer):
        syncer.sync_mkt_equ_d()

    def test_get_equ_name(self):
        df_equ = db.getDfBySql(
            "select sec_id,ticker, sec_short_name from \
            stock.equity where list_status_cd = 'L' "
        )
        ticker = "870204"
        assert (
            "沪江材料"
            == df_equ.loc[df_equ.ticker == ticker, "sec_short_name"].iloc[0]
        )

    @skip
    def test_sync_mkt_fund_day(self, syncer: StockSyncer, db: DBAdaptor):
        # syncer.sync_fund_day()
        # df = syncer.fetch_remote_fund_day_data()
        # df = pd.read_pickle(configs["cache_folder"].data + "sync_fund_day_20220204222828.pkl")
        # df['ticker'] =df['ts_code'][0:6]
        # assert df is not None and df.shape[0]>0
        # logger.info(f"获取到基金日行情行数: {df.shape[0]}")

        # rc = syncer.write_fund_day_to_db(df)
        status = db.getSyncStatusByTablename("fund_day")
        logger.info(status)
        assert status.rc
        assert status.update_time == datetime.today().date()
        assert status.comment != ""

    def test_verify_integrity_fund_day(self, db: DBAdaptor):
        df = db.getDfBySql(
            "select sec_id, ticker from stock.fund_day where trade_date='20220118'"
        )
        df_fund = db.getDfBySql(
            "select distinct ticker, sec_short_name from \
            stock.fund where list_status_cd='L'"
        )
        df_outlier = df[~df.ticker.isin(df_fund.ticker)]
        df_outlier.to_csv("/tmp/fund_outlier.csv")

    def test_sync_mkt_idx_day(self, syncer: StockSyncer, db: DBAdaptor):
        pass
        # db.getDfBySql("select ")
