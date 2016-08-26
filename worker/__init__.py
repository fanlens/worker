#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some tools to work with the data"""


class ProgressCallback(object):
    def __init__(self, task):
        self.task = task

    def __call__(self, *_, **kwargs):
        self.task.update_state(state='PROGRESS', meta=kwargs)
