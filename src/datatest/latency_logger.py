from threading import Lock, Thread
import time
import torch
import numpy as np
import json
from pprint import pprint
from pathlib import Path


class SingletonMeta(type):
    """
    This is a thread-safe implementation of the LatencyLogger Singleton.
    Code template is taken from: https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LatencyLogger(metaclass=SingletonMeta):
    latencies: dict = {}
    _lock: Lock = Lock()

    @staticmethod
    def clear():
        with LatencyLogger._lock:
            LatencyLogger.latencies = {}

    @staticmethod
    def append(sequence_name: str | int, delta_time: float):
        r"""
        Append the latency and whether past predictions were altered to the logger.
        Params:
            - sequence_name (str | int): The name of the sequence or chunk.
            - delta_time (float): The time taken for the operation in milliseconds.
        """
        with LatencyLogger._lock:
            LatencyLogger.latencies.setdefault(sequence_name, []).append(delta_time)

    @staticmethod
    def median(first_n_frames_to_skip: int) -> float:
        all_latencies = []
        for k, v in LatencyLogger.latencies.items():
            all_latencies.extend(v[first_n_frames_to_skip:])
        return float(np.median(all_latencies))

    @staticmethod
    def mean(first_n_frames_to_skip: int) -> float:
        all_latencies = []
        for k, v in LatencyLogger.latencies.items():
            all_latencies.extend(v[first_n_frames_to_skip:])
        return float(np.mean(all_latencies))

    @staticmethod
    def std(first_n_frames_to_skip: int) -> float:
        all_latencies = []
        for k, v in LatencyLogger.latencies.items():
            all_latencies.extend(v[first_n_frames_to_skip:])
        return float(np.std(all_latencies))

    @staticmethod
    def min(first_n_frames_to_skip: int) -> float:
        all_latencies = []
        for k, v in LatencyLogger.latencies.items():
            all_latencies.extend(v[first_n_frames_to_skip:])
        return float(np.min(all_latencies))

    @staticmethod
    def max(first_n_frames_to_skip: int) -> float:
        all_latencies = []
        for k, v in LatencyLogger.latencies.items():
            all_latencies.extend(v[first_n_frames_to_skip:])
        return float(np.max(all_latencies))

    @staticmethod
    def percentile(p: int, first_n_frames_to_skip: int) -> float:
        all_latencies = []
        for k, v in LatencyLogger.latencies.items():
            all_latencies.extend(v[first_n_frames_to_skip:])
        return float(np.percentile(all_latencies, p))

    @staticmethod
    def len(first_n_frames_to_skip: int) -> int:
        total_length = 0
        for k, v in LatencyLogger.latencies.items():
            total_length += len(v[first_n_frames_to_skip:])
        return total_length

    @staticmethod
    def stats(first_n_frames_to_skip) -> dict:
        return {
            "median": LatencyLogger.median(first_n_frames_to_skip),
            "P99": LatencyLogger.percentile(99, first_n_frames_to_skip),
            "P95": LatencyLogger.percentile(95, first_n_frames_to_skip),
            "mean": LatencyLogger.mean(first_n_frames_to_skip),
            "std": LatencyLogger.std(first_n_frames_to_skip),
            "min": LatencyLogger.min(first_n_frames_to_skip),
            "max": LatencyLogger.max(first_n_frames_to_skip),
            "len": LatencyLogger.len(first_n_frames_to_skip),
        }

    @staticmethod
    def export_latencies_to_json(filename: str | Path):
        r"""
        Export the latencies to a JSON file.
        Params:
            - filename (str | Path): The name of the file to save the latencies.
        """
        if not isinstance(filename, Path):
            filename = Path(filename)
        if filename.suffix != ".json":
            filename = filename.with_suffix(".json")

        if not filename.parent.exists():
            raise ValueError(f"Parent directory {filename.parent} does not exist.")

        with LatencyLogger._lock:
            with open(filename, "w") as f:
                json.dump(LatencyLogger.latencies, f, indent=4)
            print(f"Latencies saved to {filename}.")

    @staticmethod
    def export_latency_stats_to_json(filename: str | Path, first_n_frames_to_skip: int):
        r"""
        Export the latency stats to a JSON file.
        Params:
            - filename (str | Path): The name of the file to save the latencies.
            - first_n_frames_to_skip (int): The number of frames to skip for latency calculation. This is useful for ignoring the initial latency spikes with model warmups etc. Around 10-20 frames is a good number to skip.
        """
        if not isinstance(filename, Path):
            filename = Path(filename)
        if filename.suffix != ".json":
            filename = filename.with_suffix(".json")

        if not filename.parent.exists():
            raise ValueError(f"Parent directory {filename.parent} does not exist.")

        all_stats = {}
        all_stats["all"] = LatencyLogger.stats(first_n_frames_to_skip=0)
        all_stats[f"first_{first_n_frames_to_skip}_frames_skipped"] = (
            LatencyLogger.stats(first_n_frames_to_skip=first_n_frames_to_skip)
        )

        with LatencyLogger._lock:
            with open(filename, "w") as f:
                json.dump(all_stats, f, indent=4)
            print(f"Latencies saved to {filename}.")


class LogLatency:
    r"""
    This is a context manager class that logs the execution time of a code block.
    Example usage:

        ```python
        with LogLatency("sample_block"):
            time.sleep(1)  # Simulating a time-consuming task
        ```
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        # Synchronize before starting the timer if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        # Synchronize again after the code block finishes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        delta_time = end_time - self.start_time  # Execution time in seconds
        delta_time = delta_time * 1000  # Convert to milliseconds
        LatencyLogger.append(self.name, delta_time)
