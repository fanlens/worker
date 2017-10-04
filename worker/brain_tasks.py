#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`Celery` tasks related to training and usage of brain/machine learning models"""
import datetime
import itertools
import os
import uuid
from functools import lru_cache
from typing import Optional, List, Dict, Union, Iterable, Any, Tuple, NamedTuple, cast

from sqlalchemy import text as sqlalchemy_text
from sqlalchemy.orm import Session
from celery.utils.log import get_task_logger

from brain.feature.fingerprint import get_fingerprint, TFingerprint
from brain.lens import Sample, ScoredPrediction, TScoredPredictionSet, Lens, LensTrainer, MODEL_FILE_ROOT, \
    model_file_path
from common.db import get_session, insert_or_ignore
from common.db.models.activities import Data, Source, TagSet, User
from common.db.models.brain import Model, Prediction
from common.job import Space
from . import ProgressCallback, exclusive_task
from .app import app

_LOGGER = get_task_logger(__name__)


@lru_cache(maxsize=256)
def get_classifier(model_id: uuid.UUID) -> Lens:
    """
    :param model_id: the id of the `Model`
    :return: the classifier for the provided `Model`
    """
    return model_id and Lens.load_from_id(model_id)


@lru_cache(maxsize=256)
def best_model_for_source_by_id(tagset_id: int, source_id: int) -> Optional[uuid.UUID]:
    """
    Select the best model for the `TagSet` / `Source` combination
    :param tagset_id: id of the `TagSet`
    :param source_id: id of the `Source`
    :return: id of best model
    """
    with get_session() as session:
        source = session.query(Source).get(source_id)
        assert source
        model = source.models.filter_by(tagset_id=tagset_id).order_by(Model.score, Model.trained_ts).first()
        if not model:
            model = session.query(Model).filter_by(tagset_id=tagset_id).order_by(Model.score, Model.trained_ts).first()
        return model.id if model else None


@app.task
def predict_text(model_id: uuid.UUID,
                 text: str,
                 fingerprint: Optional[TFingerprint] = None,
                 created_time: Optional[datetime.datetime] = None) -> List[ScoredPrediction]:
    """
    Get a prediction for the provided text.
    :param model_id: id of the `Model` to use
    :param text: the text
    :param fingerprint: a retina fingerprint. Optional, can be computed on demand
    :param created_time: when was the text created? Optional, defaults to now
    :return: the prediction for the text
    """
    assert model_id
    assert text
    # if not is_english(text):
    #     raise ValueError('text is not in english')
    if not fingerprint:
        fingerprint = get_fingerprint(text)
    if not created_time:
        created_time = datetime.datetime.utcnow()
    classifier = get_classifier(model_id=model_id)
    prediction: TScoredPredictionSet = next(iter(classifier.predict_proba([(text, fingerprint, created_time)])))
    return prediction


@app.task
def predict_stored(model_id: uuid.UUID, data: Data) -> Dict[str, Union[int, TScoredPredictionSet]]:
    """
    Generate a prediction of a stored `Data`

    :param model_id: id of the `Model` to use
    :param data: the `Data` object
    :return: the prediction
    """
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


def grouper(iterable: Iterable[Any],
            group_size: int,
            fillvalue: Optional[Any] = None) -> Iterable[Iterable[Any]]:
    """
    Collect data into fixed-length chunks or blocks
    https://docs.python.org/3/library/itertools.html
    e.g. grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    :param iterable: the source iterable
    :param group_size: size of the groups
    :param fillvalue: appended to the end
    :return: an Iterable consisting of groups of Iterables sourced from the provided iterable
    """
    args = [iter(iterable)] * group_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def _source_id_getter(datum: Data) -> int:
    return cast(int, datum.source_id)


def group_by_source_id(data: Iterable[Data]) -> Iterable[Tuple[int, Iterable[Data]]]:
    """
    :param data: an iterable source of `Data`
    :return: grouped by the source ids of the `Data`
    """
    return cast(Iterable[Tuple[int, Iterable[Data]]], itertools.groupby(data, key=_source_id_getter))


def predict_buffered(model_id: uuid.UUID, buffer: List[Tuple[int, Sample]]) \
        -> Tuple[Iterable[Tuple[int, TScoredPredictionSet]], Union[uuid.UUID, None]]:
    """
    Perform predictions for an indexed buffer of samples.
    :param model_id: model to use
    :param buffer: list of samples with index
    :return: indexed list of predictions and the used model_id
    """
    if model_id:
        classifier = get_classifier(model_id)
        buffer_ids, buffer_data = zip(*buffer)
        predictions = classifier.predict_proba(buffer_data)
        predictions_with_id = zip(buffer_ids, predictions)
        return predictions_with_id, model_id

    _LOGGER.info("No model with id: %s", str(model_id))
    return [], None


def predict_stored_all(data: Iterable[Data], session: Session) -> None:
    """
    Add prediction to all stored `Data`
    works best if sorted by tagset id and source id
    :param data: source of `Data` objects
    :param session: the `Session` the `Data` objects are bound to
    """
    prediction_group_size = 200
    current_identifier = None
    buffer: List[Sample] = list()

    def flush_predictions() -> None:
        """flush the predictions to the database"""
        if current_identifier is None:
            buffer.clear()
            return
        _LOGGER.info("Flushing %d predicitions for model id: %s", len(buffer), str(current_identifier))
        insert_predictions, insert_model_id = predict_buffered(current_identifier, buffer)
        for insert_id, insert_prediction in insert_predictions:
            insert_or_ignore(session, Prediction(
                data_id=insert_id,
                model_id=insert_model_id,
                prediction=insert_prediction))
        _LOGGER.info("Done flushing")

    for data_id, model_id, text, translation, fingerprint, time in data:
        if model_id != current_identifier or len(buffer) >= prediction_group_size:
            flush_predictions()
            buffer.clear()
            current_identifier = model_id
        buffer.append((data_id, Sample(translation or text, fingerprint, time)))
    flush_predictions()
    session.commit()


def _train_model(user_id: int,
                 tagset_id: int,
                 source_ids: Iterable[int],
                 n_estimators: int,
                 _params: Optional[Dict[str, Any]],
                 _score: float,
                 progress: Optional[ProgressCallback] = None) -> str:
    """
    :param user_id: the creating user id
    :param tagset_id: the tagset id
    :param source_ids: the source ids
    :param n_estimators: how many estimators to use for the estimator bag
    :param progress: an optional progress callback to update state
    :param _params: don't search for params and use provided
    :param _score: provide a score for fast training mode
    :return: stringified model id
    """
    assert tagset_id and source_ids
    if not 0 < n_estimators <= 1000:
        raise ValueError('invalid estimator count: %d' % n_estimators)
    session: Session
    with get_session() as session:
        tagset = session.query(TagSet).get(tagset_id)
        sources = session.query(Source).filter(Source.id.in_(tuple(source_ids))).all()
        user = session.query(User).get(user_id)
        factory = LensTrainer(user, tagset, sources, progress=progress)
        lens = factory.train(n_estimators=n_estimators, _params=_params, _score=_score)
        return str(factory.persist(lens, session))


@exclusive_task(app, Space.BRAIN, trail=True, ignore_result=True, bind=True)
def train_model(self: app.Task,
                user_id: int,
                tagset_id: int,
                source_ids: Iterable[int],
                n_estimators: int = 10,
                _params: Optional[Dict[str, Any]] = None,
                _score: float = 0.0) -> None:
    """
    Train a model.
    :param self: bound task
    :param user_id: the creating user id
    :param tagset_id: the tagset id
    :param source_ids: the source ids
    :param n_estimators: how many estimators to use for the estimator bag
    :param _params: don't search for params and use provided
    :param _score: provide a score for fast training mode
    """
    _train_model(user_id=user_id,
                 tagset_id=tagset_id,
                 source_ids=source_ids,
                 n_estimators=n_estimators,
                 _params=_params,
                 _score=_score,
                 progress=ProgressCallback(self))


@app.task()
def maintenance() -> None:
    """
    Run maintenance job for brain: clean up db entries/model ids no longer in use
    """
    _LOGGER.info("Beginning Brain maintenance...")
    (_, _, file_id_list) = next(os.walk(MODEL_FILE_ROOT))
    file_ids = set(file_id_list)
    session: Session
    with get_session() as session:
        db_ids = set([str(model_id) for (model_id,) in session.query(Model.id)])
        missing_ids = db_ids.difference(file_ids)
        _LOGGER.warning('The following model ids are missing the model file: %s', missing_ids)
        delete_ids = file_ids.difference(db_ids)
        _LOGGER.warning('The following model ids are orphaned and will be deleted: %s', delete_ids)
        for delete_id in delete_ids:
            os.remove(model_file_path(delete_id))
    _LOGGER.info("... Done Brain maintenance")


_RetrainModels = NamedTuple('_RetrainModels', [('user_id', int), ('tagset_id', int), ('source_ids', Iterable[int]),
                                               ('trained_ts', datetime.datetime)])
_RETRAIN_MODELS_SQL = sqlalchemy_text('''
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
SELECT * FROM current_models''')


def _select_retrain_model_params() -> Iterable[_RetrainModels]:
    """
    Select all models in need for retraining.
    :return: a list of model params to retrain
    """
    with get_session() as session:
        return (_RetrainModels(**row) for row in session.execute(_RETRAIN_MODELS_SQL))


@exclusive_task(app, Space.BRAIN, trail=True, ignore_result=True, bind=True)
def retrain(self: app.Task) -> None:
    """
    Retrain outdated models in need for retraining
    :param self: bound task
    """
    _LOGGER.info("Looking for models to be retrained... ")
    for user_id, tagset_id, source_ids, _ in _select_retrain_model_params():
        _LOGGER.info(
            "\tRetraining model for user: %d, tagset: %d, sources: %s...", user_id, tagset_id, str(source_ids))
        # todo use group()
        train_model(self, user_id, tagset_id, tuple(source_ids))
        _LOGGER.info(
            "\t... Done retraining model for user: %d, tagset: %d, sources: %s", user_id, tagset_id, str(source_ids))
    _LOGGER.info("... Done retraining models")
