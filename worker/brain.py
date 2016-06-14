#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import lru_cache
from brain.tagger import Tagger, to_sample
from brain.feature.language_detect import is_english
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from brain.feature.fingerprint import get_fingerprint

from worker.celery import app

tokenizer = LemmaTokenTransformer(short_url=True)


@lru_cache(maxsize=64)
def get_tagger(model_id):
    return Tagger(model_id)


@app.task
def predict(meta, model_id='default'):
    if not all([key in meta for key in ('tokens', 'fingerprint')]):
        return None
    tagger = get_tagger(model_id)
    return tagger.predict(to_sample(meta['tokens'], meta['fingerprint']))


@app.task
def predict_text(text, model_id='default'):
    if not is_english(text):
        raise ValueError('text is not in english')
    tokens = tokenizer(text)
    fingerprint = get_fingerprint(text).positions
    tagger = get_tagger(model_id)
    return tagger.predict(to_sample(tokens, fingerprint))
