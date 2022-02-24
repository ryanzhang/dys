from datetime import date
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


class S8Strategy(BaseStrategy):
    def __init__(self):
        BaseStrategy.__init__(self)
        logger.debug("Construct S8Strategy")
        pass

    def set_mkt_timing_alg(self):
        logger.debug("Not implement yet!")

    def set_equ_pool(self):
        if self.df_equ_pool is None:
            self.default_load_pool()
        self.df_equ_pool = self.df_equ_pool.query("list_sector_cd == '1'" )
        logger.debug(f'自定义股票池{self.df_equ_pool.shape[0]}')
        return self.df_equ_pool

    def set_equd_pool(self):
        pass

    def set_select_condition(self):
        """设置选股条件字符串，条件字符串按照df.query接受的语法

        Args:
            query_str (_type_): _description_

        Returns:
        """
        self.set_equ_pool()
        self.set_equd_pool()
        self.default_load_pool()
        # Optional only for debug
        # 1个亿= 100000000
        #排除ST
        self.append_select_condition("not sec_short_name.str.startswith('*ST')")
        self.append_select_condition("not sec_short_name.str.startswith('ST')")
        self.append_select_condition("open_price > 0")
        #排除科创
        self.append_select_condition("neg_market_value < 2000000000")
        self.append_select_condition("open_price < 30")
        self.append_metric(SelectMetric("list_days", m.list_days, self.df_equ_pool))
        self.append_select_condition("list_days > 20")

    def post_hook_select_equ(self):
        self.df_choice_equd.dropna(inplace=True)
        logger.debug(
            f"post hook after 选股 has been triggered {self.df_choice_equd.shape[0]}!"
        )
        pass

    def set_rank_factors(self):
        """设置排序因子列表

        Args:
            rankfactors (list): _description_

        Returns:
        """
        db = DBAdaptor(is_use_cache=True)
        df_float = db.get_df_by_sql("select * from stock.equ_share_float")
        self.append_metric(SelectMetric("wq_alpha16", m.wq_alpha16))
        self.append_metric(SelectMetric("ntra_turnover_rate_5", m.ntra_turnover_rate, 5))
        # self.append_metric(SelectMetric("float_value_60", m.float_value, 60, df_float))
        self.append_metric(SelectMetric("float_rate_60", m.float_rate, 60, df_float))
        self.append_metric(SelectMetric("ntra_bias_6", m.ntra_bias, 6))
        self.append_metric(SelectMetric("vol_rate_5_60", m.vol_rate, 5, 60))
        self.append_metric(SelectMetric("vol_20", m.ma_vol, 20))

        self.append_rankfactor(RankFactor(name="neg_market_value", bigfirst=False, weight=5))
        self.append_rankfactor(RankFactor(name="wq_alpha16", bigfirst=True, weight=1))
        self.append_rankfactor(RankFactor(name="ntra_turnover_rate_5", bigfirst=False, weight=3))
        self.append_rankfactor(RankFactor(name="float_rate_60", bigfirst=False, weight=1))
        self.append_rankfactor(RankFactor(name="ntra_bias_6", bigfirst=False, weight=1))
        self.append_rankfactor(RankFactor(name="vol_rate_5_60", bigfirst=False, weight=1))
        self.append_rankfactor(RankFactor(name="vol_20", bigfirst=False, weight=1))

        return True

    def set_trade_model(self):
        """设置交易模型

        Args:
            rankfactors (list): _description_

        Returns:
            bool: _description_
        """
        self.trade_model = TradeModel()
        logger.debug(f"交易模型已经设定{self.trade_model}")
        return True


class TestSmall8Strategy:
    # @pytest.fixture(scope="class")
    # def db(self):
    #     return DBAdaptor()

    @pytest.fixture()
    def s8s(self):
        s8s = S8Strategy()
        s8s.debug_sample_date= date(2021,1,4)
        return s8s

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_get_rank_in_one_day(self, s8s:S8Strategy):
        s8s.set_select_condition()
        df1 = s8s.df_choice_equd
        df1_sample = df1.loc[df1.trade_date==s8s.debug_sample_date, :]
        df1_sample.to_csv("/tmp/test_s8s_select.csv")
        s8s.set_margin_offset(60)
        s8s.set_date(start_date=date(2017,1,4))
        s8s.set_rank_factors()
        s8s.rank()
        logger.debug(f"排序已经完成")
        df2 = s8s.df_choice_equd
        df2_sample = df2.loc[df2.trade_date==s8s.debug_sample_date, :]
        df2_sample.reset_index(inplace=True)
        df2_sample['ticker']=df2_sample['sec_id'].str[0:6]
        df2_sample.to_csv("/tmp/test_s8s_rank.csv")

    def test_cov_between_my_and_guoren_on_20210114(self):
        mdf = pd.read_csv("/tmp/test_s8s_rank.csv")

        msub = mdf[['ticker','sec_short_name','xrank','float_rate_60', 'neg_market_value_subrank','wq_alpha16_subrank', 'ntra_turnover_rate_5_subrank','float_rate_60_subrank', 'ntra_bias_6_subrank','vol_rate_5_60_subrank', 'vol_20_subrank']]
        gdf = pd.read_csv("/tmp/guoren-20210114-s8s.csv")
        gsub = gdf[['股票代码','总排名分','未来60日新增流通股占比', '流通市值_排名分', 'WQ_Alpha16_排名分', '中性N日换手率(5)_排名分', '未来60日新增流通股占比_排名分', '中性N日乖离率(6)_排名分', 'N日M日量比(5,60)_排名分', 'N日均成交量(20)_排名分']]
        gsub.rename(columns={'股票代码':'ticker'}, inplace=True)

        merge = pd.merge(msub,gsub, on='ticker',how='outer')
        merge['xrank'].dropna(inplace=True)
        merge['总排名分'].dropna(inplace=True)
        merge.to_csv("/tmp/mg_compare_result.csv")
        merge.corr(method='spearman').to_csv('/tmp/mg_compare_corr.csv')
        logger.debug("breakpoint")
        
