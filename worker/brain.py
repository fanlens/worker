#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import itertools
import os
import typing
import uuid
from functools import lru_cache

from celery.utils.log import get_task_logger
from sqlalchemy import text

from brain.feature.fingerprint import get_fingerprint
from brain.lens import Lens, LensTrainer, model_file_root, model_file_path
from db import get_session, Session, insert_or_ignore
from db.models.activities import Data, Source, TagSet, User
from db.models.brain import Model, Prediction
from job import Space
from . import ProgressCallback, exclusive_task
from .celery import app

logger = get_task_logger(__name__)


@lru_cache(maxsize=256)
def get_classifier(model_id: uuid.UUID) -> Lens:
    return model_id and Lens.load_from_id(model_id)


@lru_cache(maxsize=256)
def best_model_for_source_by_id(tagset_id: int, source_id: int) -> uuid.UUID:
    with get_session() as session:  # type: Session
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


def predict_buffered(model_id, buffer):
    if model_id:
        classifier = get_classifier(model_id)
        buffer_ids, buffer_data = zip(*buffer)
        # logger.error(buffer_ids)
        # logger.error(buffer_data)
        predictions = classifier.predict_proba(buffer_data)
        # logger.error(predictions)
        # logger.error(list(zip(buffer_ids, predictions)))
        return zip(buffer_ids, predictions), model_id
    else:
        logger.info("No model for with id: %s" % str(model_id))
        return [], None


def predict_stored_all(data: typing.Iterable, session: Session):
    """works best if sorted by tagset id and source id"""
    prediction_group_size = 200
    current_identifier = None
    buffer = []

    def flush_predictions():
        if current_identifier is None:
            buffer.clear()
            return
        logger.info("Flushing %d predicitions for model id: %s" % (len(buffer), str(current_identifier)))
        insert_predictions, insert_model_id = predict_buffered(current_identifier, buffer)
        for insert_id, insert_prediction in insert_predictions:
            insert_or_ignore(session, Prediction(
                data_id=insert_id,
                model_id=insert_model_id,
                prediction=insert_prediction))
        logger.info("Done flushing")

    for data_id, model_id, text, translation, fingerprint, time in data:
        if model_id != current_identifier or len(buffer) >= prediction_group_size:
            flush_predictions()
            buffer.clear()
            current_identifier = model_id
        buffer.append((data_id, (translation or text, fingerprint, time)))
    flush_predictions()
    session.commit()


def _train_model(user_id: int,
                 tagset_id: int,
                 source_ids: tuple = tuple(),
                 n_estimators: int = 10,
                 _params: dict = None,
                 _score: float = 0.0,
                 progress=None):
    assert tagset_id and source_ids
    assert n_estimators
    if not 0 < n_estimators <= 1000:
        raise ValueError('invalid estimator count: %d' % n_estimators)
    with get_session() as session:  # type: Session
        tagset = session.query(TagSet).get(tagset_id)
        sources = session.query(Source).filter(Source.id.in_(tuple(source_ids))).all()
        user = session.query(User).get(user_id)
        factory = LensTrainer(user, tagset, sources, progress=progress)
        lens = factory.train(n_estimators=n_estimators, _params=_params, _score=_score)
        return str(factory.persist(lens, session))


@exclusive_task(app, Space.BRAIN, trail=True, ignore_result=True, bind=True)
def train_model(self, user_id: int, tagset_id: int, source_ids: tuple, n_estimators: int, _params: dict, _score: float):
    _train_model(user_id=user_id,
                 tagset_id=tagset_id,
                 source_ids=source_ids,
                 n_estimators=n_estimators,
                 _params=_params,
                 _score=_score,
                 progress=ProgressCallback(self))


@app.task(bind=True)
def maintenance(self):
    logger.info("Beginning Brain maintenance...")
    (_, _, file_ids) = next(os.walk(model_file_root))
    file_ids = set(file_ids)
    with get_session() as session:  # type: Session
        db_ids = set([str(uuid) for (uuid,) in session.query(Model.id)])
        missing_ids = db_ids.difference(file_ids)
        logger.warning('The following model ids are missing the model file: %s' % missing_ids)
        delete_ids = file_ids.difference(db_ids)
        logger.warning('The following model ids are orphaned and will be deleted: %s' % delete_ids)
        for delete_id in delete_ids:
            os.remove(model_file_path(delete_id))
    logger.info("... Done Brain maintenance")


def select_retrain_model_ids() -> set:
    with get_session() as session:  # type: Session
        return session.execute(text('''
WITH modles_by_user_tagset_source_trained AS (
    SELECT model.id, model.created_by_user_id, model.tagset_id, jsonb_agg(DISTINCT src_mdl.source_id ORDER BY src_mdl.source_id) AS sources, model.trained_ts
    FROM activity.model AS model
    JOIN activity.source_model AS src_mdl ON model.id = src_mdl.model_id
    GROUP BY model.id, model.created_by_user_id, model.id, model.trained_ts
),
current_models AS (
    SELECT created_by_user_id, tagset_id, sources, max(trained_ts) AS trained_ts
    FROM modles_by_user_tagset_source_trained
    GROUP BY created_by_user_id, tagset_id, sources),
relevant_data AS (
    SELECT model.id AS model_id, dat.id AS data_id, dat.crawled_ts AS crawled_ts, model.trained_ts AS trained_ts
    FROM activity.data AS dat
    JOIN activity.language AS lang ON lang.data_id = dat.id
    JOIN activity.text AS text ON text.data_id = dat.id
    LEFT OUTER JOIN activity.translation AS trans ON trans.text_id = text.id
    JOIN activity.source_model AS sm ON sm.source_id = dat.source_id
    JOIN activity.model AS model ON model.id = sm.model_id
    WHERE (lang.language = 'en' OR trans.target_language = 'en') 
),
outdated_models AS (
    SELECT relevant_data.model_id AS id
    FROM relevant_data
    WHERE relevant_data.crawled_ts > relevant_data.trained_ts
    GROUP BY relevant_data.model_id
    HAVING count(*) > 50
    UNION
    SELECT relevant_data.model_id AS id
    FROM relevant_data
    JOIN activity.tagging AS tagging ON tagging.data_id = relevant_data.data_id
    WHERE tagging.tagging_ts > relevant_data.trained_ts
    GROUP BY relevant_data.model_id
    HAVING count(*) > 50 OR min(tagging.tagging_ts) < (now() - INTERVAL '4 hours')
)
SELECT created_by_user_id, tagset_id, sources, trained_ts
FROM modles_by_user_tagset_source_trained AS models
JOIN outdated_models AS outdated ON models.id = outdated.id
INTERSECT
SELECT * FROM current_models'''))


@exclusive_task(app, Space.BRAIN, trail=True, ignore_result=True, bind=True)
def retrain(self):
    logger.info("Looking for models to be retrained... ")
    for user_id, tagset_id, source_ids, _ in select_retrain_model_ids():
        logger.info(
            "\tRetraining model for user: %d, tagset: %d, sources: %s..." % (user_id, tagset_id, str(source_ids)))
        # todo use group()
        _train_model(user_id, tagset_id, tuple(source_ids))
        logger.info(
            "\t... Done retraining model for user: %d, tagset: %d, sources: %s" % (user_id, tagset_id, str(source_ids)))
    logger.info("... Done retraining models")


if __name__ == "__main__":
    logger.getLogger().setLevel(logger.DEBUG)
    logger.getLogger().addHandler(logger.StreamHandler())
    # retrain()


    with get_session() as session:  # type: Session
        current_user = session.query(User).get(5)
        session.query()
