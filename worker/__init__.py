#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some tools to work with the data"""

from celery import Celery
from celery.utils import gen_task_name
from job import runs_exclusive, Space
import logging

class ProgressCallback(object):
    def __init__(self, task):
        self.task = task

    def __call__(self, *_, **kwargs):
        self.task.update_state(state='PROGRESS', meta=kwargs)


def exclusive_task(app: Celery, space: Space, *task_args, **task_kwargs):
    """
    helper to create a quick exclusive run
    example:
    @exclusive_task(app, Space.WORKER, trail=True, ignore_result=True)
    def random_task():
        pass
    """

    def wrapper(fun: callable):
        @app.task(*task_args, name=gen_task_name(app, fun.__name__, fun.__module__), **task_kwargs)
        @runs_exclusive(space)
        def task(*args, **kwargs):
            logging.error('wtf' + str(args) + str(kwargs))
            return fun(*args, **kwargs)

        return task

    return wrapper
