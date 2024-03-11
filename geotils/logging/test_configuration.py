import unittest
from yacs.config import CfgNode as CN
import yaml
import os

class TestConfigManager(unittest.TestCase):
    """
    Test cases for the ConfigManager class methods.
    """

    def setUp(self):
        """
        Set up method to initialize common parameters for tests.
        """
        self.config_manager = ConfigManager(".")

        sample_config = CN()
        sample_config.model = CN()
        sample_config.model.name = "resnet"
        sample_config.model.num_layers = 50
        sample_config.optimizer = CN()
        sample_config.optimizer.type = "adam"
        sample_config.optimizer.lr = 0.001

        # Set the configuration as the cfg attribute of ConfigManager
        self.config_manager.cfg = sample_config

        # Get the configuration dictionary using get_configs method
        retrieved_config = self.config_manager.get_configs()

        # Compare the retrieved configuration with the sample configuration
        self.assertEqual(retrieved_config, sample_config)

    def test_convert_cfg_to_dict(self):
        """
        Test case to verify the convert_cfg_to_dict method of ConfigManager.
        """
        # Define a sample configuration CfgNode
        sample_cfg_node = CN()
        sample_cfg_node.model = CN()
        sample_cfg_node.model.name = "resnet"
        sample_cfg_node.model.num_layers = 50
        sample_cfg_node.optimizer = CN()
        sample_cfg_node.optimizer.type = "adam"
        sample_cfg_node.optimizer.lr = 0.001

        # Convert the sample configuration CfgNode to a dictionary
        converted_dict = self.config_manager.convert_cfg_to_dict(sample_cfg_node)

        # Define the expected dictionary
        expected_dict = {
            "model": {"name": "resnet", "num_layers": 50},
            "optimizer": {"type": "adam", "lr": 0.001}
        }

        # Compare the converted dictionary with the expected dictionary
        self.assertEqual(converted_dict, expected_dict)

    def test_save_cfg_as_yaml(self):
        """
        Test case to verify the save_cfg_as_yaml method of ConfigManager.
        """
        # Define a sample configuration dictionary
        sample_config = {
            "model": {"name": "resnet", "num_layers": 50},
            "optimizer": {"type": "adam", "lr": 0.001}
        }

        # Call the save_cfg_as_yaml method with the sample configuration dictionary
        self.config_manager.save_cfg_as_yaml(sample_config)

        # Check if the YAML file has been created
        yaml_file_path = f"{self.config_manager.config_file_path}/configs.yaml"
        self.assertTrue(os.path.exists(yaml_file_path))

        # Read the saved YAML file and check if it matches the sample configuration
        with open(yaml_file_path, "r") as yaml_file:
            saved_config = yaml.safe_load(yaml_file)
        self.assertEqual(saved_config, sample_config)

    def test_load_cfg_from_yaml(self):
        """
        Test case to verify the _load_cfg_from_yaml method of ConfigManager.
        """
        # Define a sample configuration dictionary
        sample_config = {
            "model": {"name": "resnet", "num_layers": 50},
            "optimizer": {"type": "adam", "lr": 0.001}
        }

        # Write the sample configuration to a YAML file
        yaml_file_path = f"{self.config_manager.config_file_path}/configs.yaml"
        with open(yaml_file_path, "w") as yaml_file:
            yaml.safe_dump(sample_config, yaml_file, default_flow_style=False)

        # Call the _load_cfg_from_yaml method
        loaded_config_node = self.config_manager._load_cfg_from_yaml()

        # Convert the loaded CfgNode instance to a dictionary
        loaded_config_dict = self.config_manager.convert_cfg_to_dict(loaded_config_node)

        # Check if the loaded configuration matches the sample configuration
        self.assertEqual(loaded_config_dict, sample_config)
