#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
from bs4 import BeautifulSoup

from sqlalchemy import literal
from db import DB, insert_or_ignore
from db.models.scrape import Shortener

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
    with DB().ctx() as session:
        result = insert_or_ignore(session, Shortener(**tags))
        session.commit()
        insert_id = result.inserted_primary_key and result.inserted_primary_key[0] or None
    return insert_id, tags


if __name__ == "__main__":
    logging.getLogger().addHandler(logging.StreamHandler())
    print(scrape_meta_for_url( 'http://www.miamiherald.com/news/nation-world/article159642739.html'))
    print(scrape_meta_for_url( 'http://docs.sqlalchemy.org/en/latest/core/tutorial.html#coretutorial-insert-expressions'))
    print(scrape_meta_for_url( 'https://www.nytimes.com/2017/07/07/briefing/g20-pennsylvania-station-tesla.html?rref=collection%2Fseriescollection%2Fus-morning-briefing&action=click&contentCollection=briefing&region=stream&module=stream_unit&version=latest&contentPlacement=1&pgtype=collection'))
