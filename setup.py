#!/usr/bin/python
#
# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding=utf-8
from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='kaggle-environments',
    version=get_version("kaggle_environments/__init__.py"),
    description='Kaggle Environments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kaggle',
    author_email='support@kaggle.com',
    url='https://github.com/Kaggle/kaggle-environments',
    keywords=['Kaggle'],
    entry_points={'console_scripts': [
        'kaggle-environments = kaggle_environments.main:main']},
    install_requires=["jsonschema >= 3.0.1"],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    license='Apache 2.0')
