#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import lru_cache
from brain.tagger import TaggerFactory
from brain.feature.language_detect import is_english
from brain.feature.fingerprint import get_fingerprint

from worker.celery import app


@lru_cache(maxsize=64)
def get_tagger(model_id):
    return TaggerFactory().name(model_id).tagger


@app.task
def predict(text, fingerprint, model_id='default'):
    tagger = get_tagger(model_id)
    return tagger.predict((text, fingerprint))


@app.task
def predict_text(text, model_id='default'):
    if not is_english(text):
        raise ValueError('text is not in english')
    fingerprint = get_fingerprint(text).positions
    tagger = get_tagger(model_id)
    return tagger.predict((text, fingerprint))


@app.task
def train_model(user_id, tagset_id, sources=None, name=None, params=None):
    factory = (TaggerFactory()
               .user_id(user_id)
               .tagset(tagset_id)
               .sources(sources)
               .name(name)
               .params(params)
               .train()
               .persist())
    return factory.tagger.name
