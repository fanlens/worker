#!/usr/bin/env python
# -*- coding: utf-8 -*-

from worker.celery import app
from worker.meta_pipeline import meta_pipeline
import crawler.process


class ProgressCallback(object):
    def __init__(self, task):
        self.task = task

    def __call__(self, *_, **kwargs):
        self.task.update_state(state='PROGRESS', meta=kwargs)


@app.task(bind=True, trail=True)
def crawl_page(self, page, since=None, include_extensions='comments'):
    process = crawler.process.facebook_page_process(page, since=since, include_extensions=include_extensions,
                                                    progress=ProgressCallback(self))
    process.start()
    return meta_pipeline.delay()
