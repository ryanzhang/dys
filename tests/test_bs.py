from datetime import date, datetime, timedelta
import os
import sys
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger
import warnings

from dys import *
import pandas as pd
from dys.domain import SelectMetric

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail

warnings.simplefilter(action="ignore", category=FutureWarning)


class MyStrategy(BaseStrategy):
    def __init__(self, start_date: str = None, end_date: str = None):
        if start_date is not None and end_date is not None:
            BaseStrategy.__init__(self, start_date, end_date)
        else:
            BaseStrategy.__init__(self)
        logger.debug("Construct MyStrategy")

    # def set_cache_folder(self, path):
    #     """For unit test only

    #     Args:
    #         path (_type_): _description_
    #     """        
    #     self.config.data_folder = path
    #     self._BaseStrategy__mk_folder()

    def set_mkt_timing_alg(self) -> bool:
        return True

    def set_equ_pool(self) -> bool:
        db = DBAdaptor(is_use_cache=True)

        df = db.get_df_by_sql(
            "select * from stock.equity where exchange_cd = 'XSHG'"
        )
        self.df_equ_pool = df

        df_equd = self.select_equd_by_equ_pool()

        logger.debug(
            f"自定义股票池已经设定成功{self.df_equ_pool.shape[0]} {self.df_equd_pool.shape[0]}"
        )
        return df_equd

    def set_metrics(self):
        reset_cache = False
        self.append_metric(
            SelectMetric("ma5_vol_rate", m.vol_rate, 5),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("list_days", m.list_days, self.df_equ_pool),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("list_days", m.list_days, self.df_equ_pool),
            reset_cache=reset_cache,
        )
        self.append_metric(
            SelectMetric("wq_alpha16", m.wq_alpha16), reset_cache=reset_cache
        )

    def set_select_condition(self):
        """设置选股条件字符串，条件字符串按照df.query接受的语法

        Args:
            query_str (_type_): _description_

        Returns:
        """
        #set_equ_pool should be explicit calling
        # self.set_equ_pool()
        # Optional only for debug
        # 1个亿= 100000000
        # 排除将要退市股票
        self.append_select_condition("not sec_short_name.str.startswith('退市')")
        self.append_select_condition("not sec_short_name.str.endswith('退')")
        self.append_select_condition("not sec_short_name.str.startswith('ST')")
        self.append_select_condition(
            "not sec_short_name.str.startswith('*ST')"
        )
        self.append_select_condition("open_price > 0")

        # 排除科创
        self.append_select_condition("neg_market_value < 2000000000")
        self.append_select_condition("open_price < 30")
        self.append_select_condition("list_days > 20")

    def post_hook_select_equ(self):
        self.df_choice_equd.dropna(inplace=True)
        logger.debug(
            f"post hook after 选股 has been triggered {self.df_choice_equd.shape[0]}!"
        )
        pass

    def set_rank_factors(self) -> bool:
        """设置排序因子列表

        Args:
            rankfactors (list): _description_

        Returns:
            bool: _description_
        """
        self.append_rankfactor(
            RankFactor(name="wq_alpha16", bigfirst=True, weight=1)
        )
        return True

    def set_trade_model(self) -> bool:
        """设置交易模型

        Args:
            rankfactors (list): _description_

        Returns:
            bool: _description_
        """
        self.trade_model = TradeModel(
            xperiod=1,
            xtiming=1,
            bench_num=5,
            unit_ideal_pos_pct=15,
            unit_pos_pct_tolerance=30,
            mini_unit_buy_pct=1,
            buy_fee_rate=0.3 / 1000,
            sale_fee_rate=2 / 1000,
        )
        self.trade_model.append_buy_criterial("rank<=8")
        self.trade_model.append_buy_criterial("chg_pct>-0.098")
        self.trade_model.append_sale_criterial(
            "sec_short_name.str.startswith('*ST')"
        )
        self.trade_model.append_buy_criterial("sale_days>=3")
        self.trade_model.append_sale_criterial("ma5_vol_rate>3")
        self.trade_model.append_sale_criterial("rank >= 34")
        self.trade_model.append_notsale_criterial("chg_pct > 0.098")
        logger.debug(f"交易模型已经设定{self.trade_model}")
        return True


class MyETFStrategy(BaseStrategy):
    def __init__(self):
        BaseStrategy.__init__(self, trade_type=1)
        logger.debug("Construct MyETFStrategy")


class TestBaseStrategy:
    # 加载一年的数据，并加载120天的margin 用于计算指标
    @pytest.fixture()
    def ms(self):
        ms = MyStrategy("20200709", "20211231")
        # Modify the config.data_folder
        ms.set_metric_folder(sys.path[-1] + "/target")
        ms.set_metrics()
        return ms

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_module_name(self):
        logger.debug(f"{self.__class__.__name__}")
        pass

    def test_using_default_construct_ok(self):
        ms = MyStrategy()
        assert ms is not None
        assert ms.df_equ_pool is not None
        assert ms.df_equd_pool is not None
        assert ms.df_choice_equd is not None
        assert ms.config is not None
        assert ms.config.data_folder is not None
        assert os.path.exists(ms.config.data_folder + "/cache/6f258.parquet")

    def test_customize_date_construck_ok(self):
        start_date = "20210104"
        end_date = "20211229"
        logger.debug(
            f"Construct customize date from {start_date} to {end_date}"
        )
        ms = MyStrategy(start_date, end_date)
        assert ms is not None
        assert ms.df_equ_pool is not None
        assert ms.df_equd_pool is not None
        assert ms.df_choice_equd is not None

        logger.debug(f"current equ pool size:{ms.df_equ_pool.shape[0]}")
        logger.debug(f"current equd pool size:{ms.df_equd_pool.shape[0]}")

    def test_select_equd_by_daterange_without_customize_equ_pool(
        self, ms: MyStrategy
    ):

        df1 = ms.select_equd_by_daterange(date(2020, 7, 9))
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2020, 7, 9)
        df2 = ms.select_equd_by_daterange(date(2021, 7, 6), date(2021, 8, 6))
        assert df2 is not None
        assert df2["trade_date"].iloc[0] == date(2021, 7, 6)
        assert df2["trade_date"].iloc[-1] == date(2021, 8, 6)

    def test_select_equd_by_daterange_with_customize_equ_pool(self, ms: MyStrategy):
        # set_equ_pool custimze the equ poo to only XSHG
        ms.set_equ_pool()
        df1: pd.DataFrame = ms.select_equd_by_daterange(date(2021, 7, 9))
        assert df1 is not None
        assert df1["trade_date"].iloc[0] == date(2021, 7, 9)
        # 有退市的股票
        n = df1["ticker"].nunique()
        assert n <= ms.df_equ_pool.shape[0]
        assert (
            df1.loc[df1.sec_id.str.endswith("XSHG"), :].shape[0]
            == df1.shape[0]
        )
        assert df1["ticker"].nunique() <= ms.df_equ_pool.shape[0]

    def test__add_metric_column(self, ms: MyStrategy):
        reset_cache = True
        ms.append_metric(
            SelectMetric("MOM20_close_price", m.momentum, 20, "close_price")
        )
        # ms.append_metric(
        #     SelectMetric("float_value_60", m.float_value, 60),
        #     reset_cache=reset_cache,
        # )
        # ms.append_metric(
        #     SelectMetric("float_rate_60", m.float_rate, 60),
        #     reset_cache=reset_cache,
        # )
        df = ms.df_choice_equd
        assert "MOM20_close_price" in df.columns
        assert os.path.exists(
            ms.config.data_folder + "metrics/MOM20_close_price.paquet"
        )
        logger.debug(df)

    def test_select_equ_by_expression(self, ms: MyStrategy):
        ms.select_equd_by_daterange(date(2021, 1, 4))
        # ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms.append_select_condition("close_price < 20")
        df = ms.df_choice_equd
        assert df is not None
        rowcount = df.shape[0]
        assert rowcount == df.loc[df["close_price"] < 20, :].shape[0]

        logger.debug(df[["trade_date", "ticker", "close_price"]])

    def test_select_by_date(self, ms: MyStrategy):
        ms.select_equd_by_daterange(date(2021, 3, 4))
        df = ms.df_choice_equd

        assert df.iloc[0].trade_date == pd.to_datetime(date(2021, 3, 4))
        logger.debug(df)

    def test_ranking_1_factor(self, ms: MyStrategy):
        ms.append_metric(
            SelectMetric("MOM20_close_price", m.momentum, 20, "close_price")
        )

        ms.select_equd_by_daterange(date(2022, 1, 5))

        ms.append_rankfactor(
            RankFactor(name="MOM20_close_price", bigfirst=True, weight=1)
        )
        ms.post_hook_select_equ()
        df = ms.rank()
        assert "rank" in df.columns
        assert "MOM20_close_price_subrank" in df.columns
        df.to_csv("/tmp/test_ranking.csv")
        logger.debug(df.columns)
        logger.debug(
            df[
                [
                    "trade_date",
                    "ticker",
                    "MOM20_close_price",
                    "rank",
                    "MOM20_close_price_subrank",
                ]
            ]
        )

    def test_ranking_1_factor_small_first(self, ms: MyStrategy):
        ms.set_equ_pool()
        ms.select_equd_by_daterange(date(2022, 1, 4))

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
        ms.append_metric(SelectMetric("mom20", m.momentum, 20, "close_price"))
        ms.set_select_condition()
        ms.select_equd_by_daterange(date(2022, 1, 4))
        ms.post_hook_select_equ()

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
        the_date = date(2021, 1, 5)
        df = ms.get_choice_equ_by_date(the_date)
        assert df is not None
        logger.debug(f"{the_date} 共选出 {df.shape[0]}")

    def test_get_choice_equd_metrics_by_list(self, ms: MyStrategy):
        ms.append_metric(
            SelectMetric("MOM20_close_price", m.momentum, 20, "close_price")
        )

        directory = sys.path[-1] + "/resources/teststrategy"
        output_directory = sys.path[-1] + "/target"
        # logger.debug(f"directory: {sys.path}")
        os.makedirs(output_directory, exist_ok=True)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            obf = os.path.join(output_directory, "buy_" + filename)
            osf = os.path.join(output_directory, "sale_" + filename)
            # checking if it is a file
            if os.path.isfile(f):
                logger.debug(f"Start to process{f}")

                choice_equ = pd.read_csv(f, dtype={"ticker": object})

                assert choice_equ is not None
                assert choice_equ.shape[0] == 2
                assert choice_equ.shape[1] == 10
                choice_equ["trade_date"] = choice_equ["买入日期"]

                df = ms.get_choice_equ_metrics_by_list(choice_equ)

                df.to_csv(obf, encoding="GBK")
                logger.debug(f"Export buy moment metrics in {obf}")
                choice_equ["trade_date"] = choice_equ["卖出日期"]
                df = ms.get_choice_equ_metrics_by_list(choice_equ)
                df.to_csv(osf, encoding="GBK")
                logger.debug(f"Export sale moment metrics in {osf}")

    def test_generate_position_mfst(self, ms: MyStrategy):
        start_date = date(2021, 1, 4)
        end_date = date(2021, 12, 31)
        ms.set_equ_pool()
        ms.set_select_condition()
        ms.select_equd_by_daterange(start_date=start_date, end_date=end_date)
        ms.set_rank_factors()
        ms.rank()
        ms.set_trade_model()
        ms.generate_position_mfst()
        mfst = ms.get_fmt_position_mfst()
        max_drawback = ms.get_history_max_drawdown()
        max_roi = ms.get_history_max_roi()
        final_roi = ms.get_roi_by_date()
        assert mfst is not None
        mfst.to_csv("/tmp/test_ms_position_mfst.csv")
        ms.df_sale_mfst.to_csv("/tmp/test_ms_sale_mfst.csv")
        # logger.debug(mfst)
        logger.info(f"最终收益:{final_roi}")
        logger.info(f"最大收益:{max_roi}")
        logger.info(f"最大回撤:{max_drawback}")
        # ms.T.drop_duplicates().T
        ms.df_position_mfst.to_parquet("/tmp/position_mfst_2010.parquet")
        ms.df_sale_mfst.to_parquet("/tmp/sale_mfst_2010.parquet")

    @skip
    def test_get_trade_mfst_by_date(self, ms: MyStrategy):
        the_date = date(2021, 1, 5)
        df = ms.get_trade_mfst_by_date(the_date)
        assert df is not None
        logger.debug(f"{the_date} 共选出 {df.shape[0]}")

    @skip
    def test_roi_mfst(self, ms: MyStrategy):

        ms.set_equ_pool()
        ms.set_equd_pool()
        ms.set_select_equ_condition("close_price<20")
        ms.select_equd_by_daterange(date(2022, 1, 4))
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
