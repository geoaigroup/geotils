# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'geotils')))
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
sys.path.insert(0, os.path.abspath('../..'))

project = 'Geotils'
copyright = '2024, GEOAI'
author = 'GEOAI'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx_rtd_theme', 'sphinx.ext.napoleon']


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'pages/reference-architecture', 'some/other/file.txt']
autodoc_mock_imports = ['icecream', 'segmentation_models_pytorch', 'tmp', 'utils', 'pystac', 'imantics', 'timm', 'supermercado', 'cv2','imantics', 'simplification', 'mercantile', 'colorama', 'albumentations', 'torchvision', 'color_map', 'shapely', 'PIL', 'tabulate', 'ToBeChecked', 'Dataset', 'fcntl', 'yacs','skimage', 'rasterio', 'tqdm', 'matplotlib', 'sklearn', 'geopandas', 'pandas', 'numpy', 'torch', 'yaml']




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

