#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import crawler.process
from . import ProgressCallback
from .celery import app
from .meta_pipeline import meta_pipeline

from db import DB
from db.models.facebook import FacebookCommentEntry


@app.task(bind=True, trail=True)
def crawl_page(self, page, since=None, include_extensions='comments'):
    process = crawler.process.facebook_page_process(page,
                                                    since=since,
                                                    include_extensions=include_extensions,
                                                    progress=ProgressCallback(self))
    process.start()
    return meta_pipeline.delay()


def comment_from_json(json: dict) -> dict:
    # todo created time
    # todo more generic approach?! decouple from facebook
    # todo remove post_id = 0 from both the comments and the posts table
    return FacebookCommentEntry(
        post_id=0,
        id=json['id'],
        data={
            'id': json['id'],
            'from': {
                'id': json['user']['id'],
                'name': json['user']['name']
            },
            'message': json['message']
        },
        meta={
            'page': json['page']
        })


@app.task(trail=True)
def add_comment(comment_json):
    with DB().ctx() as session:
        try:
            comment_entry = comment_from_json(comment_json)
        except KeyError as err:
            logging.exception("could not store comment, doesn't conform to specification")
            return None
        session.add(comment_entry)
        session.commit()
        return meta_pipeline.delay()


@app.task(trail=True)
def add_comment_bulk(bulk):
    with DB().ctx() as session:
        try:
            comment_entries = [comment_from_json(json) for json in bulk['comments']]
        except KeyError as err:
            logging.exception("could not store comment, doesn't conform to specification")
            return None

        session.add_all(comment_entries)
        session.commit()
        return meta_pipeline.delay()
