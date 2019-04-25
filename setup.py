#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

# Check if current package is compatible with current python installation (> 3.6)
if sys.version_info < (3, 6, 0):
  raise Exception("RadAnalyze requires python 3.6 or later")

NAME = "RadAnalyze"
DESCRIPTION = "Radiomics analysis using machine learning methods"
KEYWORDS = "Radiomics analysis"
LICENSE = 'MIT License'
AUTHOR = "Zhengting Cai"
AUTHOR_EMAIL = "jety2858@163.com, caizhengting2858@gmail.com"
URL = "https://github.com/Happykelee/Radiomics-Analyze"
VERSION = "0.1.0"
REQUIRES = ['numpy',
            'pandas',
            'matplotlib',
            'scipy',
            'scikit-learn>=0.20.0']

setup(
    name = NAME,
    url = URL,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    version = VERSION,
    packages = find_packages(),
    description = DESCRIPTION ,
    license = LICENSE,
    keywords=KEYWORDS,
    include_package_data = True,
    platforms = 'any',
    install_requires = REQUIRES,
)
