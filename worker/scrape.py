#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup

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
    return tags


if __name__ == "__main__":
    print(scrape_meta_for_url(
        'http://www.miamiherald.com/news/nation-world/article159642739.html'))
