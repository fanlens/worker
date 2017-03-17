#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import dateutil.parser

from celery import group
from datetime import datetime

from brain.feature.language_detect import language_detect
from brain.feature.fingerprint import get_fingerprints
from db import DB
from db.models.users import User
from db.models.brain import Job
from db.models.activities import Data, Lang, Language, Type, Text, Time, Fingerprint
from utils.buffered import Buffered
from utils.simple_utc import simple_utc

from .celery import app
from .brain import predict_stored_all

text_extractors = {
    Type.facebook: lambda data: data.data['message'],
    Type.twitter: lambda data: data.data['text'],
    Type.generic: lambda data: data.data['text'],
}


def _extract_text(data: Data) -> Data:
    text = text_extractors[data.source.type](data).replace('\b', '').encode('ascii', 'ignore').decode('utf-8', 'ignore')
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
    lang = language_detect(data.text.text)
    data.language = Language(language=Lang[lang])
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


class FingerprintHandler(object):
    def __init__(self, session):
        self._session = session

    def __call__(self, buffer):
        texts = [entry.text.text for entry in buffer]
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
        entries = session.query(Data).filter(Data.language.has(language='en') & (Data.fingerprint == None))
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
                    | add_fingerprint.s()
                    | add_prediction.s()
                    | unlock.s(str(job.id)))()
        else:
            logging.info('Already a meta pipeline running ...')


