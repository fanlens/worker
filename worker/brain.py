#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import lru_cache
from brain.tagger import Tagger, to_sample

from worker.celery import app


@lru_cache(maxsize=64)
def get_tagger(model_id):
    return Tagger(model_id)


@app.task
def predict(meta, model_id='default'):
    if not all([key in meta for key in ('tokens', 'fingerprint')]):
        return None
    tagger = get_tagger(model_id)
    return tagger.predict(to_sample(meta['tokens'], meta['fingerprint']))
