from datetime import date
import numpy as np
import pytest
import pandas as pd
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.domain import SelectMetric
from dys.metric import m

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


class TestMetrics:
    @pytest.fixture()
    def dfs(self):
        db = DBAdaptor(is_use_cache=True)
        df = db.get_df_by_sql(
            "select * from stock.mkt_equ_day where trade_date >= '20200709' and trade_date<='20211231' order by id"
        )
        df_no_sus = df.loc[df.open_price > 0, :]
        # 把量能有关的 停牌日设置为空，预期失真不如排出
        # df.loc[df.open_price==0, 'turnover_vol']= np.nan
        # df.loc[df.open_price==0, 'turnover_rate']= np.nan
        # df.loc[df.open_price==0, 'turnover_value']= np.nan
        # df.loc[df.open_price==0, 'deal_amount']= np.nan
        # assert df is not None
        return (df_no_sus, df)

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        # db= DBAdaptor(is_use_cache=True)
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_ps_pit(self, dfs):
        df = dfs[0]
        # df_with_sus=dfs[1]
        # sm = SelectMetric("chg_pct_60", m.n_chg_pct, 60, df_with_sus)
        # df_metric = sm.apply(df, sm.name, sm.args)
        # df = df.join(df_metric)

        # ps_pit是每股收益指标统称 具体指标列名需参考文档
        sm = SelectMetric(f"ps_pit", m.ps_pit)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[df[sm.name] == 0, :]

        # assert df[sm.name].notna().all()
        logger.debug(df_metric)
        logger.debug(df_sample_null_metric)

    def test_neg_share_incr(self, dfs):
        df = dfs[0]
        # df_with_sus=dfs[1]
        # sm = SelectMetric("chg_pct_60", m.n_chg_pct, 60, df_with_sus)
        # df_metric = sm.apply(df, sm.name, sm.args)
        # df = df.join(df_metric)

        N = 60
        sm = SelectMetric(f"neg_share_incr_{N}", m.neg_share_incr, N, dfs[1])
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[df[sm.name] == 0, :]

        # assert df[sm.name].notna().all()
        logger.debug(df_metric)
        logger.debug(df_sample_null_metric)

    def test_suspend_in(self, dfs):
        df = dfs[0]
        # df_with_sus=dfs[1]
        # sm = SelectMetric("chg_pct_60", m.n_chg_pct, 60, df_with_sus)
        # df_metric = sm.apply(df, sm.name, sm.args)
        # df = df.join(df_metric)

        N = 5
        sm = SelectMetric(f"not_suspend_in_{N}", m.not_suspend_in, N, dfs[1])
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[df[sm.name] == 0, :]

        # assert df[sm.name].notna().all()
        logger.debug(df_metric)
        logger.debug(df_sample_null_metric)

    def test_roi_volat(self, dfs):
        df = dfs[0]
        # df_with_sus=dfs[1]
        # sm = SelectMetric("chg_pct_60", m.n_chg_pct, 60, df_with_sus)
        # df_metric = sm.apply(df, sm.name, sm.args)
        # df = df.join(df_metric)

        N = 60
        sm = SelectMetric(f"roi_volat_{N}", m.roi_volat, N)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[
            (df["ticker"] == "838030") & (df_metric[sm.name].isna()), :
        ]
        assert df_sample_null_metric.shape[0] == N - 1
        # assert df[sm.name].notna().all()
        logger.debug(df_metric)

    def test_price_aml(self, dfs):
        df = dfs[0]
        sm = SelectMetric("price_ampl", m.price_ampl)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[
            (df["ticker"] == "838030") & (df_metric[sm.name].isna()), :
        ]
        assert df_sample_null_metric.shape[0] == 0
        assert df[sm.name].notna().all()
        logger.debug(df_metric)

    def test_ma_price_aml_rate(self, dfs):
        df = dfs[0]
        sm1 = SelectMetric("price_ampl", m.price_ampl)
        df_metric = sm1.apply(df, sm1.name, sm1.args)
        assert sm1.name in df_metric.columns
        df = df.join(df_metric)

        sm2 = SelectMetric("price_ampl_rate", m.price_ampl_rate)
        df_metric = sm2.apply(df, sm2.name, sm2.args)
        assert sm2.name in df_metric.columns
        df = df.join(df_metric)

        sm3 = SelectMetric(
            "ma10_price_ampl_rate", m.ma_any, 10, "price_ampl_rate"
        )
        df_metric = sm3.apply(df, sm3.name, sm3.args)
        assert sm3.name in df_metric.columns
        df = df.join(df_metric)
        assert df[sm3.name].notna().all()
        logger.debug(df_metric)
        pass

    def test_N_chg_pct(self, dfs):
        df = dfs[0]
        df_with_sus = dfs[1]
        sm = SelectMetric("20_chg_pct", m.n_chg_pct, 20, df_with_sus)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[
            (df["ticker"] == "000002") & (df_metric[sm.name].isna()), :
        ]
        assert df_sample_null_metric.shape[0] == 20
        logger.debug(df_metric)
        pass

    def test_revers_21(self, dfs):
        df = dfs[0]
        df_with_sus = dfs[1]
        sm = SelectMetric("revers_21", m.revers_21)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[
            (df["ticker"] == "000002") & (df_metric[sm.name].isna()), :
        ]
        assert df_sample_null_metric.shape[0] == 20
        logger.debug(df_metric)
        pass

    def test_momentum(self, df: pd.DataFrame):
        sm = SelectMetric("MOM20_close_price", m.momentum, 20, "close_price")
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        df = df.join(df_metric)
        df_sample_null_metric = df.loc[
            (df["ticker"] == "000002") & (df_metric[sm.name].isna()), :
        ]
        assert df_sample_null_metric.shape[0] == 20
        logger.debug(df_metric)

    def test_list_days(self, df: pd.DataFrame):
        db = DBAdaptor()
        df_list_days = db.get_df_by_sql(
            "select * from stock.equity where list_status_cd ='L'"
        )
        sm = SelectMetric("list_days", m.list_days, df_list_days)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        # Workaround , there are missing data causing na row exist
        # assert df[sm.name].notna().all()
        logger.debug(df_metric)

    def test_wq_alpha16(self, df: pd.DataFrame):
        # db = DBAdaptor()
        sm = SelectMetric("wq_alpha16", m.wq_alpha16)
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert "wq_alpha16" in df_metric.columns
        df_sample_null_metric = df.loc[
            (df["ticker"] == "603192")
            & (df["trade_date"] == pd.to_datetime("20210104")),
            :,
        ]
        # Alpha 因子会计算5日MA，所以会有4天空白
        # assert df_sample_null_metric.shape[0] == 4
        df_sample_null_metric.to_csv("/tmp/wq_alpha16.csv")
        # logger.debug(df_metric)

    def test_ntra_turnover_rate(self, df: pd.DataFrame):
        sm = SelectMetric("ntra_turnover_rate", m.ntra_turnover_rate, 5)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert "ntra_turnover_rate" in df_metric.columns
        # assert df[sm.name].notna().all()
        logger.debug(df_metric)

    def test_bias(self, df: pd.DataFrame):
        N = 6
        sm = SelectMetric("bias", m.bias, N)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns

        df = df.join(df_metric)
        df_000001 = df[df.ticker == "000001"]
        df_000001[df_000001["bias"].isna()].shape[0] == N - 1
        df_000001.set_index("trade_date", inplace=True)
        df_000001.dropna(inplace=True)
        logger.debug(df_000001[["close_price", "bias"]])
        # df_000001.to_csv("/tmp/test_bias_000001.csv")

    def test_netra_bias(self, df: pd.DataFrame):
        sm = SelectMetric("nbias", m.ntra_bias, 6)
        df_metric = sm.apply(df, sm.name, sm.args)
        assert sm.name in df_metric.columns
        logger.debug(df_metric)

        df = df.join(df_metric)

        df_000001 = df[df.ticker == "000001"]
        df_000001.set_index("trade_date", inplace=True)
        df_000001.dropna(inplace=True)
        logger.debug(df_000001[["close_price", "nbias"]])
        # plt.plot(df_000001["bias"], label='bias')
        # plt.plot(df_000001["close_price"], label='close_price')
        # plt.show()

    def test_float_rate_n(self, dfs: pd.DataFrame):
        df = dfs[0]
        df_with_sus = dfs[1]
        # db = DBAdaptor(is_use_cache=True)
        N = 60
        # df_float = db.get_df_by_sql("select * from stock.equ_share_float")
        sm = SelectMetric(f"float_rate_{N}", m.float_rate, N, df_with_sus)
        df_metric = sm.apply(df, sm.name, sm.args)

        # df.to_csv("/tmp/test_float_rate_all.csv")

        df = df.join(df_metric)
        df_603059 = df.loc[df.ticker == "603059", :]

        assert df[sm.name].notna().all()
        assert (
            df_603059.loc[
                df_603059.trade_date == pd.to_datetime("20210105"), sm.name
            ].iloc[0]
            > 0
        )

        # df_000156.set_index("trade_date", inplace=True)
        # df_000156.dropna(inplace=True)
        # logger.debug(df_000156[["ticker", f"float_rate_{N}"]])
        # df_000156.to_csv("/tmp/test_float_rate_000156.csv")

    def test_float_value_n(self, df: pd.DataFrame):
        # db = DBAdaptor(is_use_cache=True)
        N = 90
        # df_float = db.get_df_by_sql("select * from stock.equ_share_float")
        sm = SelectMetric(f"float_rate_{N}", m.float_value, N)
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert df[sm.name].notna().all()

        df_000156 = df[df.ticker == "000156"]
        df_000156.set_index("trade_date", inplace=True)
        # df_000156.dropna(inplace=True)
        # df_000156.to_csv("/tmp/test_bias_000156.csv")

    def test_vol_nm_rate(self, df: pd.DataFrame):
        N = 5
        M = 20
        sm = SelectMetric(f"nm_vol_rate", m.vol_nm_rate, N, M)
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert sm.name in df.columns
        df_sample = df[df["ticker"] == "000001"]
        # 这里注意, 选取的000001 在2021年没有停牌，所以下面等式成立,否则应该小于M-1
        df_sample[df_sample["nm_vol_rate"].isna()].shape[0] == M - 1

    def test_vol_rate(self, df: pd.DataFrame):
        N = 5
        sm = SelectMetric(f"ma{N}_vol_rate", m.vol_rate, N)
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert sm.name in df.columns
        df_sample = df[df["ticker"] == "000586"]
        logger.debug(
            f"{df_sample.iloc[0].trade_date}{df_sample.iloc[0].sec_short_name} {df_sample}"
        )
        df_sample[df_sample[sm.name].isna()].shape[0] == N - 1

    def test_ma_turnover_rate(self, df: pd.DataFrame):
        N = 5
        sm = SelectMetric("turnover_rate_5", m.ma_turnover_rate, N)
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert sm.name in df.columns
        df_sample = df[df["ticker"] == "000001"]
        df_sample[df_sample["turnover_rate_5"].isna()].shape[0] == N - 1

    def test_ntra_turnover_rate_5(self, df: pd.DataFrame):
        N = 5
        # sm = SelectMetric(f"turnover_rate_5", m.ma_turnover_rate, N)
        # df_metric1 = sm.apply(df,sm.name, sm.args)
        # df = df.join(df_metric1 )

        sm2 = SelectMetric("ntra_turnover_rate_5", m.ntra_turnover_rate, N)
        df_metric2 = sm2.apply(df, sm2.name, sm2.args)
        df = df.join(df_metric2)
        assert sm2.name in df.columns

    def test_neg_market_amount(self, df: pd.DataFrame):
        # sm = SelectMetric(f"turnover_rate_5", m.ma_turnover_rate, N)
        # df_metric1 = sm.apply(df,sm.name, sm.args)
        # df = df.join(df_metric1 )

        sm = SelectMetric("neg_market_amount", m.neg_market_amount)
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert sm.name in df.columns
        assert df[sm.name].notna().all()

    def test_rank(self, df: pd.DataFrame):
        # sm = SelectMetric(f"turnover_rate_5", m.ma_turnover_rate, N)
        # df_metric1 = sm.apply(df,sm.name, sm.args)
        # df = df.join(df_metric1 )
        N = 5
        # sm = SelectMetric(f"turnover_rate_5", m.ma_turnover_rate, N)
        # df_metric1 = sm.apply(df,sm.name, sm.args)
        # df = df.join(df_metric1 )

        sm2 = SelectMetric("ntra_turnover_rate_5", m.ntra_turnover_rate, N)
        df_metric2 = sm2.apply(df, sm2.name, sm2.args)
        df = df.join(df_metric2)
        assert sm2.name in df.columns

        sm = SelectMetric(
            "ntra_turnover_rate_5_rank", m.rank, "ntra_turnover_rate_5", True
        )
        df_metric = sm.apply(df, sm.name, sm.args)
        df = df.join(df_metric)
        assert sm.name in df.columns
        # assert df[sm.name].notna().all()
