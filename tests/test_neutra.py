import matplotlib.pyplot as plt
import math
import pytest
import pandas as pd
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger
from sqlalchemy import column

from dys.bs import BaseStrategy
from dys.domain import RankFactor, SelectMetric
from dys.metric import m
from dys.neutralize import Neutra


given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


class TestNeutra:
    @pytest.fixture()
    def df(self):
        db = DBAdaptor(is_use_cache=True)
        df = db.get_df_by_sql(
            "select * from stock.mkt_equ_day where trade_date = '20210104'"
        )
        assert df is not None
        return df

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_ntra_apply(self, df: pd.DataFrame):
        ntra = Neutra(df["market_value"])
        df["nturnover_rate"] = ntra.apply(df["turnover_rate"])
        df["npb"] = ntra.apply(df["pb"])
        df["npe"] = ntra.apply(df["pe"])
        df["npe1"] = ntra.apply(df["pe1"])
        df["nvwap"] = ntra.apply(df["vwap"])
        df["nvwap"] = ntra.apply(df["vwap"])

        assert "nturnover_rate" in df.columns
        df.to_csv("/tmp/test_ntra_equd.csv")
        pass

    def test_ntra_apply(self, df: pd.DataFrame):
        ntra = Neutra(df["market_value"])
        df['bias'] = df['chg_pct']*100
        df["nbias"] = ntra.apply(df["bias"])

        assert "bias" in df.columns
        logger.debug(df)
        pass
    @skip
    def test_ntra_apply_with_plot(self, df: pd.DataFrame):
        ntra = Neutra(df["market_value"])

        df["nturnover_rate"] = ntra.apply(df["turnover_rate"])
        df["npb"] = ntra.apply(df["pb"])
        df["npe"] = ntra.apply(df["pe"])
        df["npe1"] = ntra.apply(df["pe1"])
        df["nvwap"] = ntra.apply(df["vwap"])

        df["tturnover_rate"] = ntra.standalize(df["turnover_rate"])
        df["tpb"] = ntra.standalize(df["pb"])
        df["tpe"] = ntra.standalize(df["pe"])
        df["tpe1"] = ntra.standalize(df["pe1"])
        df["tvwap"] = ntra.standalize(df["vwap"])

        df["pturnover_rate"] = ntra.predict(df["turnover_rate"])
        df["ppb"] = ntra.predict(df["pb"])
        df["ppe"] = ntra.predict(df["pe"])
        df["ppe1"] = ntra.predict(df["pe1"])
        df["pvwap"] = ntra.predict(df["vwap"])
        df.sort_values("market_value", inplace=True)
        df.set_index("market_value", inplace=True)

        # df.plot.scatter(x='market_value', y='')

        assert "nturnover_rate" in df.columns
        df.to_csv("/tmp/test_ntra_equd.csv")
        # for c in ['turnover_rate','pb','pe1','vwap']:
        for c in ["pe1"]:
            plt.figure()
            plt.plot(df["p" + c], label="predict_" + c, linewidth=4)
            plt.plot(df["n" + c], label="diff_" + c, linewidth=4)
            plt.plot(df["t" + c], label=c)
        plt.show()
        pass
