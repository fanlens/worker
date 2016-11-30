#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import uuid
from functools import lru_cache

from db import DB
from db.models.activities import Data, Source, TagSet
from brain.lens import Lens, LensTrainer
from brain.feature.language_detect import is_english
from brain.feature.fingerprint import get_fingerprint
from . import ProgressCallback

from .celery import app

_db = DB()


@lru_cache(maxsize=32)
def get_classifier(model_id: uuid.UUID) -> Lens:
    return Lens.load_from_id(model_id)


@app.task
def predict_text(model_id: uuid.UUID, text, fingerprint=None, created_time=None):
    assert model_id
    assert text
    # if not is_english(text):
    #     raise ValueError('text is not in english')
    if not fingerprint:
        fingerprint = get_fingerprint(text).positions
    if not created_time:
        created_time = datetime.datetime.utcnow()
    classifier = get_classifier(model_id=model_id)
    predicition = classifier.predict_proba([(text, fingerprint, created_time)])
    return list(predicition)


@app.task
def predict_strict(model_id: uuid.UUID, data: Data):
    assert data.text
    assert data.language
    assert data.fingerprint
    assert data.created_time
    classifier = get_classifier(model_id)
    assert data.source in classifier.model.sources
    return dict(id=data.id, predicition=predict_text(model_id,
                                                     data.text.text,
                                                     data.fingerprint.fingerprint,
                                                     data.time.time))


@app.task(bind=True)
def train_model(self, tagset_id: int, source_ids: tuple = tuple(), n_estimators: int = 10, params: dict = None):
    assert tagset_id and source_ids
    assert n_estimators
    if not 0 < n_estimators <= 1000:
        raise ValueError('invalid estimator count: %d' % n_estimators)
    with _db.ctx() as session:
        tagset = session.query(TagSet).get(tagset_id)
        sources = session.query(Source).filter(Source.id.in_(tuple(source_ids))).all()
    factory = LensTrainer(tagset, sources, progress=ProgressCallback(self))
    lens = factory.train(n_estimators=n_estimators, params=params)
    return str(factory.persist(lens))
