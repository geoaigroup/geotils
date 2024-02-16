#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:50:11 2021

@author: hasan
"""
#configs
from yacs.config import CfgNode as CN
import yaml

class ConfigManager:
    """
    A class to manage configurations loaded from a YAML file.

    Explanation:
        This class provides methods to access, convert, save, and load configuration
        data from a YAML file.

    Args:
        None

    Returns:
        None
    """

    def __init__(self, config_file_path):
        """
        Constructor for ConfigManager.Initializes the ConfigManager 
        with the path to the YAML configuration file.

        Args:
            config_file_path (str): The path to the YAML configuration file.

        Returns:
            None
        """
        self.config_file_path = config_file_path
        self.cfg = self._load_cfg_from_yaml()

    def get_configs(self):
        """
        Get the configuration as a dictionary.

        Explanation:
            Returns a copy of the configuration as a dictionary, useful for accessing
            individual configuration values.

        Args:
            None

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return self.cfg.clone()

    def convert_cfg_to_dict(self, cfg_node, key_list=[]):
        """
        Convert a config node to a dictionary recursively.

        Explanation:
            This method recursively converts a CfgNode instance (the internal
            representation of the config) to a regular Python dictionary.

        Args:
            cfg_node (CfgNode): The CfgNode instance to convert.
            key_list (list[str], optional): A list to track the current key path during
                recursive conversion. Defaults to an empty list.

        Returns:
            dict: The converted dictionary representation of the CfgNode.
        """
        VALID_TYPES = {tuple, list, str, int, float, bool}
        if not isinstance(cfg_node, CN):
            if type(cfg_node) not in VALID_TYPES:
                print(
                    f"Key {'.'.join(key_list)} with value {cfg_node} is not a valid type; valid types: {VALID_TYPES}"
                )
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = self.convert_cfg_to_dict(v, key_list + [k])
            return cfg_dict

    def save_cfg_as_yaml(self, config_dict):
        """
        Save the configuration dictionary as a YAML file.

        Explanation:
            Saves the provided dictionary representation of the configuration to a YAML file
            at the specified path.

        Args:
            config_dict (dict): The dictionary representing the configuration to save.

        Returns:
            None
        """
        with open(f"{self.config_file_path}/configs.yaml", "w+") as yaml_file:
            yaml.safe_dump(config_dict, yaml_file, default_flow_style=False)

    def _load_cfg_from_yaml(self):
        """
        Load the configuration from a YAML file.

        Explanation:
            This method internally loads the configuration from the specified YAML file path
            and converts it to a CfgNode instance.

        Args:
            None

        Returns:
            CfgNode: The CfgNode instance representing the loaded configuration.
        """
        with open(f"{self.config_file_path}/configs.yaml", "r") as yaml_file:
            loaded = yaml.safe_load(yaml_file)
        return CN(loaded)
