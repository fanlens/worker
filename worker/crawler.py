#!/usr/bin/env python
# -*- coding: utf-8 -*-
import crawler.process
from . import ProgressCallback
from .celery import app
from .meta_pipeline import meta_pipeline


@app.task(bind=True, trail=True)
def crawl_page(self, page, since=None, include_extensions='comments'):
    process = crawler.process.facebook_page_process(page,
                                                    since=since,
                                                    include_extensions=include_extensions,
                                                    progress=ProgressCallback(self))
    process.start()
    return meta_pipeline.delay()
