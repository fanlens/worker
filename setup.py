#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""package used for the ai components of the fanlens project"""

from setuptools import setup, find_packages

setup(
    name="fl-worker",
    version="0.1",
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fl-base',
        'fl-crawler',
        'celery[redis]',
        'msgpack-python'
    ],
)