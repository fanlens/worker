#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from celery import group

import requests
from bs4 import BeautifulSoup
from crawler.process import facebook_crawler_process
from db import get_session, Session, insert_or_ignore
from db.models.scrape import Shortener, CrawlLog, CrawlState
from sqlalchemy import text
from job import Space

from . import exclusive_task
from .celery import app


@app.task
def fetch_url_content(url):
    return requests.get(url).text


OG_TAGS = {'locale', 'type', 'title', 'description', 'url', 'site_name', 'image'}
TWITTER_TAGS = {'card', 'site', 'title', 'description', 'image', 'url'}


def find_or_none(soup: BeautifulSoup, elem, value, **property):
    tag = soup.find(elem, attrs=property)
    return tag and tag.get(value)


@app.task
def scrape_meta_for_url(url):
    html_doc = fetch_url_content(url)
    soup = BeautifulSoup(html_doc, 'html.parser')

    tags = dict(
        orig_source=find_or_none(soup, 'link', 'href', rel='original-source'),
        description=find_or_none(soup, 'meta', 'content', name='description'),
        canonical=find_or_none(soup, 'link', 'href', rel='canonical'))

    twitter_tags = dict((tag, find_or_none(soup, 'meta', 'content', name='twitter:%s' % tag))
                        for tag in TWITTER_TAGS)
    tags.update(twitter_tags)

    og_tags = dict((tag, find_or_none(soup, 'meta', 'content', property='og:%s' % tag))
                   for tag in OG_TAGS)
    tags.update(og_tags)
    with get_session() as session:  # type: Session
        result = insert_or_ignore(session, Shortener(**tags))
        session.commit()
        insert_id = result.inserted_primary_key and result.inserted_primary_key[0] or None
    return insert_id, tags


@app.task
def crawl(source_id):
    with get_session() as session:  # type: Session
        session.add(CrawlLog(source_id=source_id, state=CrawlState.START))
        session.commit()
        try:
            process = facebook_crawler_process(source_id, -60)
            process.start()
            session.add(CrawlLog(source_id=source_id, state=CrawlState.DONE))
            session.commit()
        except Exception:
            session.add(CrawlLog(source_id=source_id, state=CrawlState.FAIL))
            session.commit()


@exclusive_task(app, Space.CRAWLER, trail=True, ignore_result=True)
def recrawl():
    outdated = text('''
        SELECT * FROM (
            SELECT DISTINCT ON(src.id)
                src.id, cl.state, cl.timestamp
            FROM activity.source AS src
            LEFT OUTER JOIN activity.crawllog AS cl ON cl.source_id = src.id
            WHERE src.auto_crawl = TRUE
            ORDER BY src.id, cl.timestamp DESC
        ) AS most_recent_per_source
        WHERE timestamp IS NULL
            OR (state = 'START' AND timestamp < now() - '8 hours'::INTERVAL) -- timeout
            OR (state = 'DONE' AND timestamp < now() - '2 hours'::INTERVAL) -- normal schedule
            OR state = 'FAIL' ''')
    with get_session() as session:  # type: Session
        crawl_group = group(crawl.s(source_id) for (source_id, last_state, last_timestamp) in session.execute(outdated))
    return crawl_group()


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler())
    print(recrawl())
