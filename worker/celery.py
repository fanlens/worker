#!/usr/bin/env python
# -*- coding: utf-8 -*-

from celery import Celery
from config.db import Config

config = Config("worker")

app = Celery('fanlens',
             broker=config['broker'],
             backend=config['backend'],
             include=[
                 'worker.meta',
                 'worker.brain'
             ])

# Optional configuration, see the application user guide.
app.conf.update(
    CELERYD_LOG_FORMAT="[%(asctime)s: %(levelname)s/%(processName)s:%(funcName)s] %(message)s",
    CELERY_IGNORE_RESULT=False,
    CELERY_TRACK_STARTED=True,
    CELERY_TASK_SERIALIZER='msgpack',
    CELERY_RESULT_SERIALIZER='msgpack',
    CELERY_ACCEPT_CONTENT=['msgpack'],
    CELERY_TASK_RESULT_EXPIRES=config['task_result_expires'],
    CELERY_REDIRECT_STDOUTS=False,
    TIMEZONE='UTC',
    CELERYBEAT_SCHEDULE={
        'scheduled_meta_pipeline': {
            'task': 'worker.meta.meta_pipeline',
            'schedule': config['meta_schedule']
        },
        'scheduled_brain_maintenance': {
            'task': 'worker.brain.maintenance',
            'schedule': config['maintenance_schedule']
        }
    })

if __name__ == '__main__':
    app.start()
