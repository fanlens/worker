#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""`Celery` app reference and setup"""

from celery import Celery
from common.config import get_config

_CONFIG = get_config()

# pylint: disable=invalid-name
app = Celery('fanlens',
             broker=_CONFIG.get('CELERY', 'broker'),
             backend=_CONFIG.get('CELERY', 'backend'),
             include=[
                 'worker.meta_tasks',
                 'worker.brain_tasks',
                 'worker.scrape_tasks',
             ])

# Optional configuration, see the application user guide.
app.conf.update(
    FANLENS_META_MODULES=_CONFIG.get('WORKER', 'meta_modules').split(','),
    CELERYD_FORCE_EXECV=True,
    CELERYD_LOG_FORMAT="[%(asctime)s: %(levelname)s/%(processName)s:%(funcName)s] %(message)s",
    CELERY_IGNORE_RESULT=False,
    CELERY_TRACK_STARTED=True,
    CELERY_TASK_SERIALIZER='msgpack',
    CELERY_RESULT_SERIALIZER='msgpack',
    CELERY_ACCEPT_CONTENT=['msgpack'],
    CELERY_TASK_RESULT_EXPIRES=_CONFIG.get('CELERY', 'task_result_expires'),
    CELERY_REDIRECT_STDOUTS=False,
    TIMEZONE='UTC',
    CELERYBEAT_SCHEDULE={
        'scheduled_meta_pipeline': {
            'task': 'worker.meta_tasks.meta_pipeline',
            'schedule': _CONFIG.getint('WORKER', 'meta_schedule')
        },
        'scheduled_brain_maintenance': {
            'task': 'worker.brain_tasks.maintenance',
            'schedule': _CONFIG.getint('WORKER', 'maintenance_schedule')
        },
        'scheduled_brain_retrain': {
            'task': 'worker.brain_tasks.retrain',
            'schedule': _CONFIG.getint('WORKER', 'retrain_schedule')
        },
        'scheduled_scrape_recrawl': {
            'task': 'worker.scrape_tasks.recrawl',
            'schedule': _CONFIG.getint('WORKER', 'recrawl_schedule')
        }
    })

if __name__ == '__main__':
    app.start()
