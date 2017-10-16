"""some tools to work with the data"""
from typing import Any, Callable, TypeVar

from celery.utils import gen_task_name

from celery import Celery, Task
from common.job import runs_exclusive, Space
from common.utils.progress import ProgressCallbackBase


class ProgressCallback(ProgressCallbackBase):
    # pylint: disable=too-few-public-methods
    """Simple Helper that stores the progress of the current task int the task meta"""

    def __init__(self, task: Task) -> None:
        """:param task: which task this callback is bound to"""
        self.task = task

    def __call__(self, *_: Any, **meta: Any) -> None:
        """:param meta: information to store in tasks meta"""
        self.task.update_state(state='PROGRESS', meta=meta)


_RT = TypeVar('_RT')


def exclusive_task(app: Celery, space: Space, *task_args: Any, **task_kwargs: Any) -> Any:
    """
    helper to create a quick exclusive run
    example:
    @exclusive_task(app, Space.WORKER, trail=True, ignore_result=True)
    def random_task():
        pass

    :param app: the `Celery` app
    :param space: which postgres `Space` to run in
    :param task_args: optional arguments passed to the task wrapper
    :param task_kwargs: optional keyword arguments passed to the task wrapper
    :return: the wrapped `Celery` task
    """

    def wrapper(fun: Callable[..., _RT]) -> Any:
        # pylint: disable=missing-docstring
        @app.task(*task_args, name=gen_task_name(app, fun.__name__, fun.__module__), **task_kwargs)
        @runs_exclusive(space)
        def task(*args: Any, **kwargs: Any) -> _RT:
            return fun(*args, **kwargs)

        return task

    return wrapper
