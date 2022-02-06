
from datetime import datetime, timedelta
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger
from kupy.config import configs
from strategy.domain import BigSmallEtfRotate
from strategy.big_small_eft_rotate import BigSmallEtfRotateStrategy
import os

given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail

class TestBigSmallEtfRotateStrategy:
    @pytest.fixture(scope="class")
    def db(self):
        return DBAdaptor(is_use_cache=True)
    
    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")
        yield
        #只保留第一行数据
        db = DBAdaptor()
        db.delete_by_id_range(BigSmallEtfRotate,min=2,max=-1)
        
    
    def test_stragegy_can_run_as_daemon(self, db:DBAdaptor):
        bser = BigSmallEtfRotateStrategy()
        start_date=datetime.today().date() -timedelta(30)
        bser.run_as_daemon(start_date)


    def test_stragegy_can_run_as_daemon_but_suspend(self, db:DBAdaptor):
        bser = BigSmallEtfRotateStrategy()
        start_date=datetime.today().date() -timedelta(30)
        bser.run_as_daemon(start_date)
        #运行20天，然后suspend， 10天
        bser.set_suspend_status(True)
        start_date=datetime.today().date() -timedelta(10)
        bser.run_as_daemon(start_date)
        #suspend 5天之后，再恢复
        bser.set_suspend_status(False)
        start_date=datetime.today().date() -timedelta(5)
        bser.run_as_daemon(start_date)

    # @skip
    def test_stragegy_can_run(self, db:DBAdaptor):
        bser = BigSmallEtfRotateStrategy()
        bser.run_with_plot(datetime.today().date())
