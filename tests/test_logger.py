from kupy.logger import logger


def test_logger():
    assert logger is not None
    logger.info("This is info level message")
    logger.debug("This is debug level message")
    logger.warning("This is warning level message")
    logger.error("This is warning level message")
