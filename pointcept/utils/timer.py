# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-

import timeit, datetime, functools
from time import perf_counter
from typing import Optional


class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset the timer.
        """
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Returns:
            bool: whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self) -> None:
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        # pyre-fixme[58]: `-` is not supported for operand types `float` and
        #  `Optional[float]`.
        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        Returns:
            (float): the total number of seconds since the start/reset of the
                timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Returns:
            (float): the average number of seconds between every start/reset and
            pause.
        """
        return self.seconds() / self._count_start


class tic_toc:
    """timer with custom message"""

    def __init__(self, message="time used", end='\n'):#, color="light_yellow"):
        # assert color in termcolor.COLORS, "{} not in termcolor.COLORS".format(color)
        self.msg = message
        self.end = end
        # self.color = color

    def __enter__(self):
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        n_second = timeit.default_timer() - self.tic
        # print("{}: {}".format(termcolor.colored(self.msg, self.color), datetime.timedelta(seconds=int(n_second))), end=self.end)
        print("{}: {}".format(self.msg, datetime.timedelta(seconds=int(n_second))), end=self.end)

    def __call__(self, f):
        """supports decorator-style usage, e.g.:
        ```python
        @tic_toc("foo")
        def bar:
            pass
        ```
        https://stackoverflow.com/questions/9213600/function-acting-as-both-decorator-and-context-manager-in-python
        """
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return decorated
