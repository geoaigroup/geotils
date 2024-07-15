import geotils.logging.configuration as conf


class test_logging:
    def _init_(self):
        self.loader = conf.ConfigManager(
            r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotils_testing"
        )

    def test_convert_cfg_to_dict(self):
        global dict
        dict = self.loader.convert_cfg_to_dict(self.loader.get_configs())

    def test_convert_cfg_to_dict(self):
        self.loader.save_cfg_as_yaml(dict)
