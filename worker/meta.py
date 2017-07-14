#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import dateutil.parser
from brain.feature.fingerprint import get_fingerprints
from brain.feature.language_detect import language_detect
# from brain.feature.translate import translate
from brain.feature.translate_microsoft import translate
from db import DB
from db.models.activities import Data, Lang, Language, Type, Text, Time, Translation, Fingerprint, TagSet, SourceUser
from db.models.brain import Model, Prediction, ModelSources
from db.models.users import User
from job import Space, close_exclusive_run
from sqlalchemy import not_
from utils.buffered import Buffered
from utils.simple_utc import simple_utc

from . import exclusive_task
from .brain import predict_stored_all
from .celery import app

text_extractors = {
    Type.facebook: lambda data: data.data['message'],
    Type.twitter: lambda data: data.data['text'],
    Type.generic: lambda data: data.data['text'],
}


def _extract_text(data: Data) -> Data:
    # text = (text_extractors[data.source.type](data)
    #     .replace('\x7f', '')
    #     .replace('\b', '')
    #     .encode('ascii', 'ignore')
    #     .decode('utf-8', 'ignore'))
    text = text_extractors[data.source.type](data)
    data.text = Text(text=text)
    return data


@app.task(trail=True, ignore_result=True)
def extract_text(*_):
    logging.info('Extracting text ...')
    num = 0
    with DB().ctx() as session:
        for datum in session.query(Data).filter(Data.text == None):
            _extract_text(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    logging.info('... Done, added %d texts' % num)


time_extractors = {
    Type.facebook: lambda data: dateutil.parser.parse(data.data.get('created_time')),
    Type.twitter: lambda data: datetime.strptime(data.data['created_at'],
                                                 '%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=simple_utc),
    Type.generic: lambda data: dateutil.parser.parse(data.data.get('created_time')),
}


def _extract_time(data: Data) -> Data:
    time = time_extractors[data.source.type](data)
    data.time = Time(time=time)
    return data


@app.task(trail=True, ignore_result=True)
def extract_time(*_):
    logging.info('Extracting time ...')
    num = 0
    with DB().ctx() as session:
        for datum in session.query(Data).filter(Data.time == None):
            _extract_time(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    logging.info('... Done, added %d times' % num)


def _add_language(data: Data) -> Data:
    try:
        lang = language_detect(data.text.text)
        data.language = Language(language=Lang[lang])
    except KeyError as err:
        logging.error('couldn\'t lang detect:', data.text.text)
        logging.exception(err)
    return data


@app.task(trail=True, ignore_result=True)
def add_language(*_):
    logging.info('Adding language ...')
    num = 0
    with DB().ctx() as session:
        for datum in session.query(Data).filter((Data.text != None) & (Data.language == None)):
            _add_language(datum)
            num += 1
            if not num % 500:
                session.commit()
        session.commit()
    logging.info('... Done, added %d languages' % num)


class TranslationsHandler(object):
    def __init__(self, session):
        self._session = session

    def __call__(self, buffer):
        texts = [entry.text.text for entry in buffer]
        if not texts:
            return
        logging.info('Creating translations for %d texts' % len(texts))
        translations = translate(texts)
        for store_entry, translation in zip(buffer, translations):
            store_entry.text.translations.append(Translation(translation=translation, target_language='en'))
        logging.info('Flushing translations data, %d translations' % len(translations))
        self._session.commit()


@app.task(trail=True, ignore_result=True)
def add_translation(*_):
    logging.info('Adding translations ...')
    with DB().ctx() as session:
        entries = (session.query(Data)
                   .join(Text, Text.data_id == Data.id)
                   .filter((Text.translations == None) & not_(Data.language.has(language='en'))))
        buffered = Buffered(entries, TranslationsHandler(session), 10)
        buffered()
        logging.info('... Done translations')


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
        logging.info('Creating fingerprints for %d texts' % len(texts))
        fingerprints = [result.positions for result in get_fingerprints(texts)]
        for store_entry, fingerprint in zip(buffer, fingerprints):
            store_entry.fingerprint = Fingerprint(fingerprint=fingerprint)
        logging.info('Flushing fingerprint data, %d fingerprints' % len(fingerprints))
        self._session.commit()


@app.task(trail=True, ignore_result=True)
def add_fingerprint(*_):
    logging.info('Adding fingerprints ...')
    with DB().ctx() as session:
        entries = (session.query(Data)
                   .join(Text, Text.data_id == Data.id)
                   .outerjoin(Translation, (Translation.text_id == Text.id))
                   .filter((Data.language.has(language='en') | (Translation.target_language == 'en')) &
                           (Data.fingerprint == None)))
        buffered = Buffered(entries, FingerprintHandler(session), 500)
        buffered()
    logging.info('... Done fingerprints')


@app.task(trail=True, ignore_result=True)
def add_prediction(*_):
    logging.info('Adding predictions ...')
    with DB().ctx() as session:
        predict_stored_all(session.query(Data.id)
                           .join(SourceUser, SourceUser.source_id == Data.source_id)
                           .join(User, (SourceUser.user_id == User.id))
                           .join(TagSet, (TagSet.user_id == User.id))
                           .join(Text, Text.data_id == Data.id)
                           .outerjoin(Translation,
                                      (Translation.text_id == Text.id) & (Translation.target_language == 'en'))
                           .join(Time, Time.data_id == Data.id)
                           .join(Fingerprint, Fingerprint.data_id == Data.id)
                           .join(Model, (Model.user_id == User.id) & (Model.tagset_id == TagSet.id))
                           .join(ModelSources,
                                 (ModelSources.model_id == Model.id) & (ModelSources.source_id == Data.source_id))
                           .outerjoin(Prediction,
                                      (Prediction.data_id == Data.id) & (Prediction.model_id == Model.id))
                           .filter(Prediction.id == None)
                           .add_columns(TagSet.id, Data.source_id, Text.text, Translation.translation,
                                        Fingerprint.fingerprint, Time.time)
                           .order_by(TagSet.id, Data.source_id)  # ordered for better caching
                           .yield_per(1000),
                           session)
    logging.info('... Done')


@app.task(trail=True, ignore_result=True)
def graciously_release_exclusive_run(run):
    """will close automatically as soon as the original run is destoryed, this is to make it more clear"""
    close_exclusive_run(run)


@exclusive_task(app, Space.WORKER, trail=True, ignore_result=True)
def meta_pipeline(*_):
    logging.info('Starting full meta pipeline workers...')
    # job_spec = (group(extract_text.s(), extract_time.s())
    # | add_language.s()
    ## | add_translation.s()
    # | add_fingerprint.s()
    # | add_prediction.s()
    # | add_prediction.s())
    # return job_spec()
    extract_text()
    extract_time()
    add_language()
    # add_translation()
    add_fingerprint()
    add_prediction()
    logging.info('Done meta pipeline workers...')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    meta_pipeline()
