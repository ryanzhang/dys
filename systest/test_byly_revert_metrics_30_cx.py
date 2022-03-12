from datetime import date
import os
import sys
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

@skip
class BylyStrategy(BaseStrategy):
    def __init__(self, start_date: str = None, end_date: str = None):
        if start_date is not None and end_date is not None:
            BaseStrategy.__init__(self, start_date, end_date)
        else:
            BaseStrategy.__init__(self)
        logger.debug("Construct BylyStrategy")
        pass

    def set_mkt_timing_alg(self):
        logger.debug("Not implement yet!")

    def set_equ_pool(self):
        self.df_equ_pool = self.df_equ_pool.query("list_sector_cd == '1' or list_sector_cd=='2'")
        self.select_equd_by_equ_pool()
        logger.debug(f"自定义股票池{self.df_equ_pool.shape[0]}")

    def set_list_days_metric(self):
        self.append_metric(
            SelectMetric("list_days", m.list_days, self.df_equ_pool),
            reset_cache=False,
        )

    def set_metrics(self):
        reset_cache = False

        self.append_metric(
            SelectMetric("wq_alpha16", m.wq_alpha16), reset_cache=reset_cache
        )
        self.append_metric(
            SelectMetric("ntra_turnover_rate_5", m.ntra_turnover_rate, 5),
            reset_cache=reset_cache,
        )
        # self.append_metric(
        #     SelectMetric("neg_share_incr_60", m.neg_share_incr, 60, self.df_equd_pool_with_sus),
        #     reset_cache=reset_cache,
        # )
        # # reset_cache=True
        # self.append_metric(
        #     SelectMetric("float_rate_60", m.float_rate, 60, self.df_equd_pool_with_sus),
        #     reset_cache=reset_cache,
        # )
        self.append_metric(
            SelectMetric("ntra_bias_6", m.ntra_bias, 6),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_20", m.ntra_bias, 20),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("bias_6", m.ntra_bias, 6),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("bias_20", m.bias, 20),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("vol_rate_5_60", m.vol_nm_rate, 5, 60),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("roi_volat_60", m.roi_volat, 60),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("roi_volat_20", m.roi_volat, 20),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("roi_volat_20", m.roi_volat, 20),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("chg_pct_21", m.n_chg_pct, 21, self.df_equd_pool_with_sus),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("price_ampl", m.price_ampl),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("price_ampl_rate", m.price_ampl_rate),
            reset_cache=False,
        )        
        self.append_metric(
            SelectMetric("ma10_price_ampl_rate", m.ma_any, 10, 'price_ampl_rate'),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("vol_rate_5_20", m.vol_nm_rate, 5, 20),
            reset_cache=False,
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
            SelectMetric("ma5_turnover_value", m.ma_any, 5,'turnover_value'), reset_cache=reset_cache
        )
        self.append_metric(
            SelectMetric("wq_alpha16_rank", m.rank, "wq_alpha16", False),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_6_rank", m.rank, "ntra_bias_6", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ntra_bias_20_rank", m.rank, "ntra_bias_20", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("vol_rate_5_60_rank", m.rank, "vol_rate_5_60", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("vol_rate_5_20_rank", m.rank, "vol_rate_5_20", True),
            reset_cache=False,
        )        
        self.append_metric(
            SelectMetric("roi_volat_60_rank", m.rank, "roi_volat_60", True),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("roi_volat_20_rank", m.rank, "roi_volat_20", True),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("neg_market_value_rank", m.rank, "neg_market_value", True),
            reset_cache=False,
        )
        self.append_metric(
            SelectMetric("market_value_rank", m.rank, "market_value", True),
            reset_cache=False,
        )

        self.append_metric(
            SelectMetric("chg_pct_21_rank", m.rank, "chg_pct_21", True),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("ma10_price_ampl_rate_rank", m.rank, "ma10_price_ampl_rate", False),
            reset_cache=False,
        )        
        self.append_metric(
            SelectMetric("ma5_turnover_value_rank", m.rank, "ma5_turnover_value", False),
            reset_cache=False,
        )        


    def set_select_condition(self):
        """设置选股条件字符串，条件字符串按照df.query接受的语法

        Args:
            query_str (_type_): _description_

        Returns:
        """
        self.set_equ_pool()
        # Optional only for debug
        # 1个亿= 100000000
        # 排除ST
        # 排除将要退市股票
        self.append_select_condition(
            "not sec_short_name.str.contains(pat = '退')"
        )
        self.append_select_condition(
            "not sec_short_name.str.contains(pat = 'S')"
        )
        self.append_select_condition("open_price > 0")

        self.append_select_condition("list_days < 365")

        self.append_select_condition("list_days > 2")

        self.append_select_condition("open_price < 30")

        self.append_select_condition("chg_pct < 0.09")

        self.append_select_condition("chg_pct > -0.09")


class TestBylyStrategy:
    # @pytest.fixture(scope="class")
    # def db(self):
    #     return DBAdaptor()

    @pytest.fixture()
    def byly(self):
        byly = BylyStrategy("20210104", "20230311")
        byly.set_metric_folder(os.getcwd() + "/starget/30_cx")
        byly.set_list_days_metric()
        # 排除确定性条件
        byly.set_select_condition()
        byly.df_equd_pool = byly.df_equd_pool.query(byly.select_conditions)
        logger.debug(f"选股池总量:{byly.df_equd_pool.shape[0]}")
        byly.debug_sample_date = date(2021, 1, 4)
        byly.set_metrics()
        return byly

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_get_detail_metric_by_export_file(self, byly: BylyStrategy):
        
        directory = os.getcwd() + "/resources/byly/30_cx"
        output_directory = os.getcwd() + "/starget"
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            obf = os.path.join(output_directory, "buy_30_cx" + filename)
            # osf = os.path.join(output_directory, "sale_" + filename)
            # checking if it is a file
            if os.path.isfile(f):
                logger.debug(f"Start to process{f}")
                choice_equ = pd.read_csv(f, dtype={"ticker": object})
                choice_equ["trade_date"] = choice_equ["买入日期"]
                choice_equ.sort_values(['ticker', 'trade_date'], inplace=True)
                choice_equ.drop_duplicates(subset=['ticker', 'trade_date'],keep='first', inplace=True)
                assert choice_equ is not None
                # assert choice_equ.shape[0] == 10
                # assert choice_equ.shape[1] == 10
                try:
                    df = byly.get_choice_equ_metrics_by_list(choice_equ)
                except Exception as e:
                    logger.debug(f"{f}")
                    raise Exception(e);
                # df.to_csv(obf, encoding="GBK")
                df.to_csv(obf)
                logger.debug(f"Export buy moment metrics in {obf}")
                # choice_equ["trade_date"] = choice_equ["卖出日期"]
                # df = byly.get_choice_equ_metrics_by_list(choice_equ)
                # df.to_csv(osf, encoding="GBK")
                # df.to_csv(osf)
                # logger.debug(f"Export sale moment metrics in {osf}")
