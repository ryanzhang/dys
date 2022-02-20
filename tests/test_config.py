from kupy.config import configs
from kupy.logger import logger
def test_properties_been_load():
    assert configs is not None
    assert configs.get("log_level").data != ""
    assert configs.get("something_not_exists") is None
