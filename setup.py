#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

# Check if current package is compatible with current python installation (> 3.6)
if sys.version_info < (3, 6, 0):
  raise Exception("RadAnalyze requires python 3.6 or later")

NAME = "RadAnalyze"
DESCRIPTION = "Radiomic analysis using machine learning methods"
KEYWORDS = "Radiomic analysis"
LICENSE = 'MIT License'
AUTHOR = "Zhengting Cai"
AUTHOR_EMAIL = "jety2858@163.com, caizhengting2858@gmail.com"
URL = "https://github.com/Happykelee/the-study-of-Python/tree/master/Scripts/RadAnalyze"
VERSION = "0.1.0"
REQUIRES = ['numpy',
            'pandas',
            'matplotlib',
            'math',
            'itertools',
            'scipy',
            'sklearn>=0.20.0']

setup(
    name = NAME,
    version = VERSION,
    keywords=KEYWORDS,
    description = DESCRIPTION ,
    license = LICENSE,
    url = URL,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires = REQUIRES,
)
