from datetime import date
import os
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys import *
import pandas as pd
from dys.domain import SelectMetric

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


class BylyStrategy(BaseStrategy):
    def __init__(self):
        BaseStrategy.__init__(self)
        logger.debug("Construct BylyStrategy")
        pass

    def set_mkt_timing_alg(self):
        logger.debug("Not implement yet!")

    def set_equ_pool(self):
        self.df_equ_pool = self.df_equ_pool.query("list_sector_cd == '1'")
        self.select_equd_by_equ_pool()
        logger.debug(f"自定义股票池{self.df_equ_pool.shape[0]}")

    def set_metrics(self):
        reset_cache = False
        self.append_metric(
            SelectMetric("list_days", m.list_days, self.df_equ_pool),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("wq_alpha16", m.wq_alpha16), reset_cache=reset_cache
        )
        self.append_metric(
            SelectMetric("ntra_turnover_rate_5", m.ntra_turnover_rate, 5),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric(
                "ntra_turnover_rate_5_rank",
                m.rank,
                "ntra_turnover_rate_5",
                True,
            ),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("float_value_60", m.float_value, 60),
            reset_cache=reset_cache,
        )
        # reset_cache=True
        self.append_metric(
            SelectMetric("float_rate_60", m.float_rate, 60),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_6", m.ntra_bias, 6),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_6_rank", m.rank, "ntra_bias_6", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_20", m.ntra_bias, 20),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_20_rank", m.rank, "ntra_bias_20", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("bias_6", m.ntra_bias, 6),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("bias_6_rank", m.rank, "bias_6", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("bias_20", m.bias, 20),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("bias_20_rank", m.rank, "bias_20", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("vol_rate_5_60", m.vol_nm_rate, 5, 60),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("vol_rate_5_60_rank", m.rank, "vol_rate_5_60", True),
            reset_cache=reset_cache,
        )
        # self.append_metric(
        #     SelectMetric("vol_20", m.ma_vol, 20), reset_cache=reset_cache
        # )


class TestBylyStrategy:
    # @pytest.fixture(scope="class")
    # def db(self):
    #     return DBAdaptor()

    @pytest.fixture()
    def byly(self):
        byly = BylyStrategy()
        byly.debug_sample_date = date(2021, 1, 4)
        byly.set_metrics()
        return byly

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_get_detail_metric_by_export_file(self, byly: BylyStrategy):
        directory = "/Users/rzhang/github/ryanzhang-appdev/quant-invest/dys/tests/resources/byly"
        output_directory = "/Users/rzhang/github/ryanzhang-appdev/quant-invest/dys/tests/target"
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            obf = os.path.join(output_directory, "buy_" + filename)
            osf = os.path.join(output_directory, "sale_" + filename)
            # checking if it is a file
            if os.path.isfile(f):
                logger.debug(f"Start to process{f}")
                choice_equ = pd.read_csv(f, dtype={"ticker": object})
                assert choice_equ is not None
                assert choice_equ.shape[0] == 10
                assert choice_equ.shape[1] == 10
                choice_equ["trade_date"] = choice_equ["买入日期"]
                df = byly.get_choice_equ_metrics_by_list(choice_equ)
                # df.to_csv(obf, encoding="GBK")
                df.to_csv(obf)
                logger.debug(f"Export buy moment metrics in {obf}")
                choice_equ["trade_date"] = choice_equ["卖出日期"]
                df = byly.get_choice_equ_metrics_by_list(choice_equ)
                # df.to_csv(osf, encoding="GBK")
                df.to_csv(osf)
                logger.debug(f"Export sale moment metrics in {osf}")
