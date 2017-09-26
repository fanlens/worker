#!/usr/bin/env python
# -*- coding: utf-8 -*-

from celery import Celery
from config import get_config

config = get_config()

app = Celery('fanlens',
             broker=config.get('CELERY', 'broker'),
             backend=config.get('CELERY', 'backend'),
             include=[
                 'worker.meta',
                 'worker.brain',
                 'worker.scrape',
             ])

# Optional configuration, see the application user guide.
app.conf.update(
    FANLENS_META_MODULES=config.get('WORKER', 'meta_modules').split(','),
    CELERYD_FORCE_EXECV=True,
    CELERYD_LOG_FORMAT="[%(asctime)s: %(levelname)s/%(processName)s:%(funcName)s] %(message)s",
    CELERY_IGNORE_RESULT=False,
    CELERY_TRACK_STARTED=True,
    CELERY_TASK_SERIALIZER='msgpack',
    CELERY_RESULT_SERIALIZER='msgpack',
    CELERY_ACCEPT_CONTENT=['msgpack'],
    CELERY_TASK_RESULT_EXPIRES=config.get('CELERY', 'task_result_expires'),
    CELERY_REDIRECT_STDOUTS=False,
    TIMEZONE='UTC',
    CELERYBEAT_SCHEDULE={
        'scheduled_meta_pipeline': {
            'task': 'worker.meta.meta_pipeline',
            'schedule': config.getint('WORKER', 'meta_schedule')
        },
        'scheduled_brain_maintenance': {
            'task': 'worker.brain.maintenance',
            'schedule': config.getint('WORKER', 'maintenance_schedule')
        },
        'scheduled_brain_retrain': {
            'task': 'worker.brain.retrain',
            'schedule': config.getint('WORKER', 'retrain_schedule')
        },
        'scheduled_scrape_recrawl': {
            'task': 'worker.scrape.recrawl',
            'schedule': config.getint('WORKER', 'recrawl_schedule')
        }
    })

if __name__ == '__main__':
    app.start()
