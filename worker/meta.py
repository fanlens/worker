#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime

import dateutil.parser
from celery.utils.log import get_task_logger
from sqlalchemy import not_

from brain.feature.fingerprint import get_fingerprints
from brain.feature.language_detect import language_detect
from brain.feature.translate import translate
from db import get_session, Session
from db.models.activities import Data, Fingerprint, Lang, Language, SourceUser, TagSetUser, Text, Time, Translation, \
    Type, SourceFeature
from db.models.brain import Model, ModelSources, ModelUser, Prediction
from db.models.users import User
from job import Space, close_exclusive_run
from utils.buffered import Buffered
from utils.simple_utc import simple_utc
from . import exclusive_task
from .brain import predict_stored_all
from .celery import app

logger = get_task_logger(__name__)

text_extractors = {
    Type.facebook: lambda data: data.data['message'],
    Type.twitter: lambda data: data.data['text'],
    Type.twitter_dm: lambda data: data.data['message_create']['message_data']['text'],
    Type.generic: lambda data: data.data['text'],
}


def _extract_text(data: Data) -> Data:
    # text = (text_extractors[data.source.type](data)
    #     .replace('\x7f', '')
    #     .replace('\b', '')
    #     .encode('ascii', 'ignore')
    #     .decode('utf-8', 'ignore'))
    text = text_extractors[Type(data.source.type)](data)
    data.text = Text(text=text)
    return data


@app.task(trail=True, ignore_result=True)
def extract_text(*_):
    logger.info('Extracting text ...')
    num = 0
    with get_session() as session:  # type: Session
        for datum in session.query(Data).filter(Data.text == None):
            _extract_text(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    logger.info('... Done, added %d texts' % num)


time_extractors = {
    Type.facebook: lambda data: dateutil.parser.parse(data.data.get('created_time')),
    Type.twitter: lambda data: datetime.strptime(data.data['created_at'],
                                                 '%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=simple_utc),
    Type.twitter_dm: lambda data: datetime.utcfromtimestamp(int(data['created_timestamp'])),
    Type.generic: lambda data: dateutil.parser.parse(data.data.get('created_time')),
}


def _extract_time(data: Data) -> Data:
    time = time_extractors[Type(data.source.type)](data)
    data.time = Time(time=time)
    return data


@app.task(trail=True, ignore_result=True)
def extract_time(*_):
    logger.info('Extracting time ...')
    num = 0
    with get_session() as session:  # type: Session
        for datum in session.query(Data).filter(Data.time == None):
            _extract_time(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    logger.info('... Done, added %d times' % num)


def _add_language(data: Data) -> Data:
    try:
        lang = language_detect(data.text.text)
        data.language = Language(language=Lang[lang])
    except KeyError as err:
        logger.error('couldn\'t lang detect:', data.text.text)
        logger.exception(err)
    return data


@app.task(trail=True, ignore_result=True)
def add_language(*_):
    logger.info('Adding language ...')
    num = 0
    with get_session() as session:  # type: Session
        for datum in session.query(Data).filter((Data.text != None) & (Data.language == None)):
            _add_language(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    logger.info('... Done, added %d languages' % num)


class TranslationsHandler(object):
    def __init__(self, session):
        self._session = session

    def __call__(self, buffer):
        texts = [entry.text.text for entry in buffer]
        if not texts:
            return
        logger.info('Creating translations for %d texts' % len(texts))
        translations = translate(texts)
        for store_entry, translation in zip(buffer, translations):
            if translation:
                store_entry.text.translations.append(Translation(translation=translation, target_language='en'))
        logger.info('Flushing translations data, %d translations' % len(translations))
        self._session.commit()


@app.task(trail=True, ignore_result=True)
def add_translation(*_):
    logger.info('Adding translations ...')
    with get_session() as session:  # type: Session
        entries = (session.query(Data)
                   .join(SourceFeature, SourceFeature.source_id == Data.source_id)
                   .join(Text, Text.data_id == Data.id)
                   .filter((Text.translations == None) &
                           (SourceFeature.feature == 'translate') &
                           not_(Data.language.has(language='en'))))
        buffered = Buffered(entries, TranslationsHandler(session), 10)
        buffered()
        logger.info('... Done translations')


class FingerprintHandler(object):
    def __init__(self, session):
        self._session = session

    @staticmethod
    def _gettext(entry: Data):
        if entry.language.language == 'en':
            return entry.text.text

        english_translation = entry.text.translations.filter(
            Translation.target_language == 'en').one_or_none()  # type: Translation
        if english_translation:
            return english_translation.translation

    def __call__(self, buffer):
        texts = [self._gettext(entry) or "unknown" for entry in buffer]
        if not texts:
            return
        logger.info('Creating fingerprints for %d texts' % len(texts))
        fingerprints = [result.positions for result in get_fingerprints(texts)]
        for store_entry, fingerprint in zip(buffer, fingerprints):
            store_entry.fingerprint = Fingerprint(fingerprint=fingerprint)
        logger.info('Flushing fingerprint data, %d fingerprints' % len(fingerprints))
        self._session.commit()


@app.task(trail=True, ignore_result=True)
def add_fingerprint(*_):
    logger.info('Adding fingerprints ...')
    with get_session() as session:  # type: Session
        entries = (session.query(Data)
                   .join(Text, Text.data_id == Data.id)
                   .outerjoin(Translation, (Translation.text_id == Text.id))
                   .filter((Data.language.has(language='en') | (Translation.target_language == 'en')) &
                           (Data.fingerprint == None)))
        buffered = Buffered(entries, FingerprintHandler(session), 500)
        buffered()
    logger.info('... Done fingerprints')


@app.task(trail=True, ignore_result=True)
def add_prediction(*_):
    logger.info('Adding predictions ...')
    with get_session() as session:  # type: Session
        predict_stored_all(session.query(Data.id)
                           .join(SourceUser, SourceUser.source_id == Data.source_id)
                           .join(User, (SourceUser.user_id == User.id))
                           .join(TagSetUser, (TagSetUser.user_id == User.id))
                           .join(Text, Text.data_id == Data.id)
                           .outerjoin(Translation,
                                      (Translation.text_id == Text.id) & (Translation.target_language == 'en'))
                           .join(Time, Time.data_id == Data.id)
                           .join(Fingerprint, Fingerprint.data_id == Data.id)
                           .join(Model, Model.tagset_id == TagSetUser.tagset_id)
                           .join(ModelUser, (ModelUser.model_id == Model.id) & (ModelUser.user_id == User.id))
                           .join(ModelSources,
                                 (ModelSources.model_id == Model.id) & (ModelSources.source_id == Data.source_id))
                           .outerjoin(Prediction,
                                      (Prediction.data_id == Data.id) & (Prediction.model_id == Model.id))
                           .filter(Prediction.id == None)
                           .add_columns(Model.id, Text.text, Translation.translation,
                                        Fingerprint.fingerprint, Time.time)
                           .order_by(Model.id)  # ordered for better caching
                           .yield_per(1000),
                           session)
    logger.info('... Done')


@app.task(trail=True, ignore_result=True)
def graciously_release_exclusive_run(run):
    """will close automatically as soon as the original run is destoryed, this is to make it more clear"""
    close_exclusive_run(run)


@exclusive_task(app, Space.WORKER, trail=True, ignore_result=True)
def meta_pipeline(*_):
    logger.info('Starting full meta pipeline workers...')
    modules = [
        ('text', extract_text),
        ('time', extract_time),
        ('language', add_language),
        ('translation', add_translation),
        ('fingerprint', add_fingerprint),
        ('prediction', add_prediction),
    ]
    for module_key, module_function in modules:
        # assure correct order
        if module_key in app.conf['FANLENS_META_MODULES']:
            module_function()

    logger.info('Done meta pipeline workers...')


if __name__ == "__main__":
    meta_pipeline()
