#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime

from functools import lru_cache
# from brain.tagger import TaggerFactory
from brain.feature.language_detect import is_english
from brain.feature.fingerprint import get_fingerprint
from . import ProgressCallback

from .celery import app


@lru_cache(maxsize=32)
def get_tagger(model_id):
    return TaggerFactory().name(model_id).tagger


@app.task
def predict(text, fingerprint=None, created_time=None, model_id='default', key_by:str=None):
    if not is_english(text):
        raise ValueError('text is not in english')
    if not fingerprint:
        fingerprint = get_fingerprint(text).positions
    if not created_time:
        created_time = datetime.datetime.utcnow()

    tagger = get_tagger(model_id)
    predicition = tagger.predict((text, fingerprint, created_time))
    if key_by is not None:
        return dict(key=key_by, prediction=predicition)
    else:
        return predicition


@app.task(bind=True)
def train_model(self, user_id, tagset_id, sources=tuple(), name=None, params=None):
    factory = (TaggerFactory(progress=ProgressCallback(self))
               .user_id(user_id)
               .tagset(tagset_id)
               .sources(sources)
               .name(name or self.request.id)
               .params(params)
               .train()
               .persist())
    return factory.tagger.name
