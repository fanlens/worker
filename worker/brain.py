#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import datetime
import uuid
import typing
from functools import lru_cache

from db import DB, Session, insert_or_ignore
from db.models.activities import Data, Source, TagSet
from db.models.brain import Model, Prediction
from brain.lens import Lens, LensTrainer
from brain.feature.language_detect import is_english
from brain.feature.fingerprint import get_fingerprint
from . import ProgressCallback

from .celery import app

_db = DB()


@lru_cache(maxsize=32)
def get_classifier(model_id: uuid.UUID) -> Lens:
    return model_id and Lens.load_from_id(model_id)


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
    prediction = list(classifier.predict_proba([(text, fingerprint, created_time)]))[0]
    return prediction


@app.task
def predict_stored(model_id: uuid.UUID, data: Data):
    assert data.text
    assert data.language
    assert data.fingerprint
    assert data.created_time
    classifier = get_classifier(model_id)
    assert classifier.model.sources.filter_by(id=data.source_id).one_or_none()
    return dict(id=data.id, predicition=predict_text(model_id,
                                                     data.text.text,
                                                     data.fingerprint.fingerprint,
                                                     data.time.time))


@lru_cache(64)
def best_model_for_source_by_id(tagset_id: int, source_id: int) -> uuid.UUID:
    with _db.ctx() as session:
        source = session.query(Source).get(source_id)
        assert source
        model = source.models.filter_by(tagset_id=tagset_id).order_by(Model.score, Model.trained_ts).first()
        return model and model.id


def grouper(iterable: typing.Iterable[typing.Any],
            n: int, fillvalue: typing.Any = None) -> typing.Iterable[typing.Iterable[typing.Any]]:
    """Collect data into fixed-length chunks or blocks"""
    # https://docs.python.org/3/library/itertools.html
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def group_by_source_id(data: typing.Iterable[Data]) -> typing.Iterable[typing.Tuple[int, typing.Iterable[Data]]]:
    return itertools.groupby(data, key=lambda datum: datum.source.id)


def predict_stored_all(tagset_id: int, data: typing.Iterable[Data], session: Session):
    """for best performance data should be sorted by their source_id"""
    prediction_group_size = 100
    commit_group_size = 500
    for commit_group in grouper(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(
                    zip(classifier.predict_proba(
                        [(datum.text.text, datum.fingerprint.fingerprint, datum.time.time)
                         for datum in prediction_group if datum]),
                        prediction_group,
                        itertools.repeat(classifier.model.id))
                    for prediction_group, classifier in zip(
                        grouper(source_group, prediction_group_size),
                        itertools.repeat(get_classifier(best_model_for_source_by_id(tagset_id, source_id))))
                    if classifier)
                for source_id, source_group in group_by_source_id(data)), commit_group_size, (None, None, None)):
        for prediction, datum, model_id in commit_group:
            prediction and insert_or_ignore(session, Prediction(data_id=datum.id, model_id=model_id, prediction=prediction))
        session.commit()


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


if __name__ == "__main__":
    print(best_model_for_source_by_id(1))
    print(best_model_for_source_by_id(2))
