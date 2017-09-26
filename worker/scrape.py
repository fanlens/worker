#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from celery import group
from celery.utils.log import get_task_logger
from sqlalchemy import text

from crawler.process import facebook_crawler_process
from db import get_session, Session, insert_or_ignore
from db.models.scrape import Shortener, CrawlLog, CrawlState
from job import Space
from . import exclusive_task
from .celery import app

logger = get_task_logger(__name__)


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
    logger.info("Crawling source: %d..." % source_id)
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
    logger.info("... Done crawling source: %d" % source_id)


@exclusive_task(app, Space.CRAWLER, trail=True, ignore_result=True)
def recrawl():
    logger.info("Starting recrawl...")
    outdated_sql = text('''
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
        outdated = list(session.execute(outdated_sql))
        logger.info("Recrawling outdated sources: " + str(outdated))
        crawl_group = group(crawl.s(source_id) for (source_id, last_state, last_timestamp) in outdated)
        logger.info("... Done recrawl")
        return crawl_group()


if __name__ == "__main__":
    print(recrawl())
