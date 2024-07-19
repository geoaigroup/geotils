import unittest
import geotils.logging.configuration as conf


class TestLogging(unittest.TestCase):
    def _init_(self):
        self.loader = conf.ConfigManager(
            r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotils_testing"
        )
        print()

    def test_convert_cfg_to_dict(self):
        global dict
        dict = self.loader.convert_cfg_to_dict(self.loader.get_configs())

    def test_convert_cfg_to_dict(self):
        self.loader.save_cfg_as_yaml(dict)


if __name__ == "__main__":
    unittest.main
