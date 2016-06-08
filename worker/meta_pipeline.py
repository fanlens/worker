#!/usr/bin/env python
# -*- coding: utf-8 -*-

from brain.feature.language_detect import language_detect
from brain.feature.lemma_tokenizer import LemmaTokenTransformer
from brain.feature.fingerprint import get_fingerprints
from db import DB, DBMapper, flag_modified, modifying
from db.models.facebook import FacebookCommentEntry
from utils.buffered import Buffered
from worker.celery import app


@modifying(keys=['meta'])
def _add_language(entry: FacebookCommentEntry) -> FacebookCommentEntry:
    lang = language_detect(entry.data['message'].replace('\b', ''))
    entry.meta['lang'] = lang
    return entry


@app.task(trail=True)
def add_language(*_):
    mapper = DBMapper(FacebookCommentEntry)
    mapper.filter = (FacebookCommentEntry.meta['lang'].astext == None)
    mapper.each(_add_language)


tokenizer = LemmaTokenTransformer(short_url=True)


@modifying(keys=['meta'])
def _add_tokens(entry: FacebookCommentEntry) -> FacebookCommentEntry:
    tokens = tokenizer(entry.data['message'].replace('\b', ''))
    entry.meta['tokens'] = tokens
    return entry


@app.task(trail=True)
def add_tokens(*_):
    mapper = DBMapper(FacebookCommentEntry)
    mapper.filter = ((FacebookCommentEntry.meta['lang'].astext == 'en') & (FacebookCommentEntry.meta['tokens'] == None))
    mapper.each(_add_tokens)


class FingerprintHandler(object):
    def __init__(self, session):
        self._session = session

    def __call__(self, buffer):
        texts = [entry.data['message'].replace('\b', '') for entry in buffer]
        if not texts:
            return
        fingerprints = [result.positions for result in get_fingerprints(texts)]
        for store_entry, fingerprint in zip(buffer, fingerprints):
            store_entry.meta['fingerprint'] = fingerprint
            flag_modified(store_entry, 'meta')
        self._session.commit()


@app.task(trail=True)
def add_fingerprint(*_):
    with DB().ctx() as session:
        entries = session.query(FacebookCommentEntry).filter(
            (FacebookCommentEntry.meta['lang'].astext == 'en') & (FacebookCommentEntry.meta['fingerprint'] == None))
        buffered = Buffered(entries, FingerprintHandler(session), 500)
        buffered()


@app.task(trail=True)
def meta_pipeline():
    return (add_language.s() | add_tokens.s() | add_fingerprint.s())()
