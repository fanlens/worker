#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import itertools
import logging
import os
import typing
import uuid
from functools import lru_cache

from brain.feature.fingerprint import get_fingerprint
from brain.lens import Lens, LensTrainer, model_file_root, model_file_path
from db import DB, Session, insert_or_ignore
from db.models.activities import Data, Source, TagSet
from db.models.brain import Model, Prediction
from job import Space, close_exclusive_run

from . import ProgressCallback, exclusive_task
from .celery import app

_db = DB()


@lru_cache(maxsize=256)
def get_classifier(model_id: uuid.UUID) -> Lens:
    return model_id and Lens.load_from_id(model_id)


@lru_cache(maxsize=256)
def best_model_for_source_by_id(tagset_id: int, source_id: int) -> uuid.UUID:
    with _db.ctx() as session:
        source = session.query(Source).get(source_id)
        assert source
        model = source.models.filter_by(tagset_id=tagset_id).order_by(Model.score, Model.trained_ts).first()
        if not model:
            model = session.query(Model).filter_by(tagset_id=tagset_id).order_by(Model.score, Model.trained_ts).first()
        return model and model.id


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


def grouper(iterable: typing.Iterable[typing.Any],
            n: int, fillvalue: typing.Any = None) -> typing.Iterable[typing.Iterable[typing.Any]]:
    """Collect data into fixed-length chunks or blocks"""
    # https://docs.python.org/3/library/itertools.html
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def group_by_source_id(data: typing.Iterable[Data]) -> typing.Iterable[typing.Tuple[int, typing.Iterable[Data]]]:
    return itertools.groupby(data, key=lambda datum: datum.source.id)


def predict_buffered(tagset_id, source_id, buffer):
    model_id = best_model_for_source_by_id(tagset_id, source_id)
    if model_id:
        classifier = get_classifier(model_id)
        buffer_ids, buffer_data = zip(*buffer)
        # logging.error(buffer_ids)
        # logging.error(buffer_data)
        predictions = classifier.predict_proba(buffer_data)
        # logging.error(predictions)
        # logging.error(list(zip(buffer_ids, predictions)))
        return zip(buffer_ids, predictions), model_id
    else:
        logging.info("No model for tagset: %d, source: %d" % (tagset_id, source_id))
        return [], None


def predict_stored_all(data: typing.Iterable, session: Session):
    """works best if sorted by tagset id and source id"""
    prediction_group_size = 200
    current_identifier = (None, None)
    buffer = []

    def flush_predictions():
        buffer_tagset_id, buffer_source_id = current_identifier
        if buffer_tagset_id is None or buffer_source_id is None:
            buffer.clear()
            return
        logging.info("Flushing %d predicitions for tagset: %d, source_id: %d" % (
            len(buffer), buffer_tagset_id, buffer_source_id))
        predictions, model_id = predict_buffered(buffer_tagset_id, buffer_source_id, buffer)
        for insert_id, insert_prediction in predictions:
            insert_or_ignore(session, Prediction(data_id=insert_id, model_id=model_id, prediction=insert_prediction))
        logging.info("Done flushing")

    for data_id, tagset_id, source_id, text, translation, fingerprint, time in data:
        if (tagset_id, source_id) != current_identifier or len(buffer) >= prediction_group_size:
            flush_predictions()
            buffer.clear()
            current_identifier = (tagset_id, source_id)
        buffer.append((data_id, (translation or text, fingerprint, time)))
    flush_predictions()
    session.commit()


@exclusive_task(app, Space.BRAIN, trail=True, ignore_result=True, bind=True)
def train_model(self, tagset_id: int, source_ids: tuple = tuple(), n_estimators: int = 10, _params: dict = None,
                _score: float = 0.0):
    assert tagset_id and source_ids
    assert n_estimators
    if not 0 < n_estimators <= 1000:
        raise ValueError('invalid estimator count: %d' % n_estimators)
    with _db.ctx() as session:
        tagset = session.query(TagSet).get(tagset_id)
        sources = session.query(Source).filter(Source.id.in_(tuple(source_ids))).all()
        factory = LensTrainer(tagset, sources, progress=ProgressCallback(self))
        lens = factory.train(n_estimators=n_estimators, _params=_params, _score=_score)
        return str(factory.persist(lens, session))


@app.task(bind=True)
def maintenance(self):
    logging.info("Beginning Brain maintenance...")
    (_, _, file_ids) = next(os.walk(model_file_root))
    file_ids = set(file_ids)
    with _db.ctx() as session:
        db_ids = set([str(uuid) for (uuid,) in session.query(Model.id)])
        missing_ids = db_ids.difference(file_ids)
        logging.warning('The following model ids are missing the model file: %s' % missing_ids)
        delete_ids = file_ids.difference(db_ids)
        logging.warning('The following model ids are orphaned and will be deleted: %s' % delete_ids)
        for delete_id in delete_ids:
            os.remove(model_file_path(delete_id))
    logging.info("... Done Brain maintenance")


if __name__ == "__main__":
    print(best_model_for_source_by_id(1))
    print(best_model_for_source_by_id(2))
