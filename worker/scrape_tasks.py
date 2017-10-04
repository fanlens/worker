#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`Celery` tasks related to crawling/scraping of web information"""
from typing import Tuple, Dict, Optional

import requests
from bs4 import BeautifulSoup
from celery import group
from celery.utils.log import get_task_logger
from sqlalchemy import text

from common.db import get_session, insert_or_ignore
from common.db.models.scrape import Shortener, CrawlLog, CrawlState
from common.job import Space
from crawler.process import facebook_crawler_process
from . import exclusive_task
from .app import app

_LOGGER = get_task_logger(__name__)


@app.task
def fetch_url_content(url: str) -> str:
    """
    :param url: url to fetch from
    :return: the urls text as string
    """
    return requests.get(url).text


_OG_TAGS = {'locale', 'type', 'title', 'description', 'url', 'site_name', 'image'}
_TWITTER_TAGS = {'card', 'site', 'title', 'description', 'image', 'url'}


def find_or_none(soup: BeautifulSoup, elem: str, value: str, **attrs: str) -> Optional[str]:
    """
    extract a tags value from an HTML page
    e.g. find_or_none(soup, 'link', 'href', rel='original-source')
    will find the 'href' value stored in the link tag element with attribute rel='original-source'
    :param soup: the parser instance for the HTML page
    :param elem: the tag elements name
    :param value: the attribute name to extract the value from
    :param attrs: additional attributes for parsing
    :return: the extracted value
    """
    tag = soup.find(elem, attrs=attrs)
    tag_value: Optional[str] = tag.get(value) if tag else None
    return tag_value


@app.task
def scrape_meta_for_url(url: str) -> Tuple[int, Dict[str, Optional[str]]]:
    """
    scrape all relevant meta data (facebook opengraph, twitter, etc.) for the url
    :param url: url to scrape from
    """
    html_doc = fetch_url_content(url)
    soup = BeautifulSoup(html_doc, 'html.parser')

    tags: Dict[str, Optional[str]] = dict(
        orig_source=find_or_none(soup, 'link', 'href', rel='original-source'),
        description=find_or_none(soup, 'meta', 'content', name='description'),
        canonical=find_or_none(soup, 'link', 'href', rel='canonical'))

    twitter_tags = dict((tag, find_or_none(soup, 'meta', 'content', name='twitter:%s' % tag))
                        for tag in _TWITTER_TAGS)
    tags.update(twitter_tags)

    og_tags = dict((tag, find_or_none(soup, 'meta', 'content', property='og:%s' % tag))
                   for tag in _OG_TAGS)
    tags.update(og_tags)
    with get_session() as session:
        result = insert_or_ignore(session, Shortener(**tags))
        session.commit()
        insert_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
    return insert_id, tags


@app.task
def crawl(source_id: int) -> None:
    """
    Crawl the provided `Source`
    :param source_id: id of the `Source`
    """
    _LOGGER.info("Crawling source: %d...", source_id)
    with get_session() as session:
        session.add(CrawlLog(source_id=source_id, state=CrawlState.START))
        session.commit()
        try:
            process = facebook_crawler_process(source_id, -60)
            process.start()
            session.add(CrawlLog(source_id=source_id, state=CrawlState.DONE))
            session.commit()
        except Exception:  # pylint: disable=broad-except
            session.add(CrawlLog(source_id=source_id, state=CrawlState.FAIL))
            session.commit()
    _LOGGER.info("... Done crawling source: %d", source_id)


@exclusive_task(app, Space.CRAWLER, trail=True, ignore_result=True)
def recrawl() -> None:
    """
    Recrawl outdated sources.
    Outdated is defined as:
        * sources that weren't crawled yet
        * sources successfully crawled over 2 hours ago
        * sources whose crawling task was started over 8 hours ago
        * sources whose last crawl failed
    """
    _LOGGER.info("Starting recrawl...")
    outdated_sql = text('''
        SELECT * FROM (
            SELECT DISTINCT ON(src.id)
                src.id, cl.state, cl.timestamp
            FROM activity.source AS src
            JOIN activity.source_feature as src_feature ON src_feature.source_id = src.id
            LEFT OUTER JOIN activity.crawllog AS cl ON cl.source_id = src.id
            WHERE src_feature.feature = 'auto_crawl'
            ORDER BY src.id, cl.timestamp DESC
        ) AS most_recent_per_source
        WHERE timestamp IS NULL
            OR (state = 'START' AND timestamp < now() - '8 hours'::INTERVAL) -- timeout
            OR (state = 'DONE' AND timestamp < now() - '2 hours'::INTERVAL) -- normal schedule
            OR state = 'FAIL' ''')
    with get_session() as session:
        outdated = list(session.execute(outdated_sql))
        _LOGGER.info("Recrawling outdated sources: %s", str(outdated))
        crawl_group = group(crawl.s(source_id) for (source_id, last_state, last_timestamp) in outdated)
        crawl_group()
        _LOGGER.info("... Done recrawl")


if __name__ == "__main__":
    print(recrawl())
