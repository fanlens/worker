# coding=utf-8

from worker.meta import meta_pipeline

if __name__ == "__main__":
    print([v for v in meta_pipeline.delay().collect()])
