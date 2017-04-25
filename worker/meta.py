#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import dateutil.parser
from brain.feature.language_detect import language_detect
from brain.feature.translate import translate
from brain.feature.fingerprint import get_fingerprints
from celery import group
from db import DB
from db.models.activities import Data, Lang, Language, Type, Text, Time, Translation, Tagging, Fingerprint
from db.models.brain import Job
from db.models.users import User
from sqlalchemy import not_
from utils.buffered import Buffered
from utils.simple_utc import simple_utc

from .brain import predict_stored_all
from .celery import app

text_extractors = {
    Type.facebook: lambda data: data.data['message'],
    Type.twitter: lambda data: data.data['text'],
    Type.generic: lambda data: data.data['text'],
}


def _extract_text(data: Data) -> Data:
    text = text_extractors[data.source.type](data).replace('\x7f', '').replace('\b', '').encode('ascii',
                                                                                                'ignore').decode(
        'utf-8', 'ignore')
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
        # todo remove condition
        entries = (session.query(Data)
                   .join(Tagging, Tagging.data_id == Data.id)
                   .join(Text, Text.data_id == Data.id)
                   .filter((Text.translations == None) &
                           not_(Data.language.has(language='en')) &
                           (Data.source_id == 9) &
                           (Tagging.tag_id == 300)))
        # entries = session.query(Data).filter(not_(Data.language.has(language='en')))
        # buffered = Buffered(entries, TranslationsHandler(session), 20)
        buffered = Buffered(entries, TranslationsHandler(session), 2)
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
        texts = [self._gettext(entry) for entry in buffer]
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
                   .join(Translation, (Translation.text_id == Text.id) & (Translation.target_language == 'en'))
                   .filter((Data.language.has(language='en') | (Translation != None)) &
                           (Data.fingerprint == None)))  # todo: possible bug at "Translation != None"
        buffered = Buffered(entries, FingerprintHandler(session), 500)
        buffered()
    logging.info('... Done fingerprints')


@app.task(trail=True, ignore_result=True)
def add_prediction(*_):
    logging.info('Adding predictions ...')
    with DB().ctx() as session:
        for user in session.query(User):
            for tagset in user.tagsets:
                predict_stored_all(tagset.id,
                                   user.data
                                   .filter((Data.text != None) & (Data.fingerprint != None) &
                                           (Data.time != None) & (Data.prediction == None))
                                   .order_by(Data.source_id),  # ordered for better caching
                                   session)
    logging.info('... Done')


@app.task(trail=True, ignore_result=True)
def unlock(self, job_id: str):
    logging.info('Unlocking meta pipeline...')
    with DB().ctx() as session:
        crawl_user = session.query(User).filter_by(email="crawler@fanlens.io").one()
        crawl_user.jobs.filter_by(id=job_id).delete()
        session.commit()
    logging.info('... Done')


@app.task(trail=True, ignore_result=True)
def meta_pipeline():
    with DB().ctx() as session:
        crawl_user = session.query(User).filter_by(email="crawler@fanlens.io").one()
        if not crawl_user.jobs.all():
            crawl_user.jobs.append(Job())
            session.commit()
            job = crawl_user.jobs.first()
            logging.info('Starting meta pipeline ...')
            return (group(extract_text.s(), extract_time.s())
                    | add_language.s()
                    # | add_translation.s()
                    | add_fingerprint.s()
                    | add_prediction.s()
                    | unlock.s(str(job.id)))()
        else:
            logging.info('Already a meta pipeline running ...')
