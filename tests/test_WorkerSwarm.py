import logging
from swimfunction import loggers
import numpy
import time
from swimfunction.context_managers.WorkerSwarm import WorkerSwarm
WorkerSwarm.num_allowed_workers = 10

LOGGER = loggers.get_console_logger(__name__, level=logging.ERROR)

def get_task_fn(arr, i):
    def fn():
        arr[i] = True
        time.sleep(0.5)
    return fn

def get_multiply_fn(arr, i, amount, sleep=0):
    def fn():
        arr[i] = arr[i] * amount
        time.sleep(sleep)
    return fn

def get_sqrt_fn(arr, i, sleep=0):
    def fn():
        arr[i] = numpy.sqrt(arr[i])
        time.sleep(sleep)
    return fn

def get_append_fn(arr, amount, sleep=0):
    def fn():
        time.sleep(sleep)
        arr.append(amount)
    return fn

def test_WorkerSwarm_nonparallel():
    jobs = []
    answer = numpy.arange(100)
    with WorkerSwarm(LOGGER) as swarm:
        for i in range(len(answer)):
            swarm.add_task(get_append_fn(jobs, i, sleep=numpy.random.rand()*0.05), parallel=False)
    assert numpy.all(numpy.isclose(jobs, answer))

def test_WorkerSwarm_sequential():
    jobs = numpy.full(20, 5, dtype=float)
    answer = numpy.sqrt(jobs.copy()) * 2
    with WorkerSwarm(LOGGER) as outer_swarm:
        with WorkerSwarm(LOGGER) as inner_swarm:
            for i in range(len(jobs)):
                inner_swarm.add_task(get_sqrt_fn(jobs, i)) # inner swarm always has higher priority
        for i in range(len(jobs)):
            outer_swarm.add_task(get_multiply_fn(jobs, i, 2))
    assert numpy.all(jobs == answer)

def test_WorkerSwarm_priority_sequential():
    jobs = numpy.full(20, 5, dtype=float)
    answer = numpy.sqrt(jobs.copy() * 2)
    with WorkerSwarm(LOGGER) as outer_swarm:
        with WorkerSwarm(LOGGER) as inner_swarm:
            for i in range(10):
                inner_swarm.add_task(get_multiply_fn(jobs, i, 2, sleep=0.2)) # inner swarm always has higher priority
            for i in range(len(jobs)):
                outer_swarm.add_task(get_sqrt_fn(jobs, i))
            for i in range(10, len(jobs)):
                inner_swarm.add_task(get_multiply_fn(jobs, i, 2))
    assert numpy.all(jobs == answer)

def test_WorkerSwarm_priority_sequential_with_nonparallel():
    jobs_sequential = []
    answer_sequential = numpy.arange(10)
    jobs = numpy.full(20, 5, dtype=float)
    answer = numpy.sqrt(jobs.copy() * 2)
    with WorkerSwarm(LOGGER) as outer_swarm:
        with WorkerSwarm(LOGGER) as inner_swarm:
            for i in range(10):
                inner_swarm.add_task(get_multiply_fn(jobs, i, 2, sleep=0.2)) # inner swarm always has higher priority
            for i in range(len(jobs)):
                outer_swarm.add_task(get_sqrt_fn(jobs, i))
            for i in range(10, len(jobs)):
                inner_swarm.add_task(get_multiply_fn(jobs, i, 2))
            for i in range(10):
                outer_swarm.add_task(get_append_fn(jobs_sequential, i), parallel=False)
    assert numpy.all(jobs == answer)
    assert numpy.all(jobs_sequential == answer_sequential)

def test_WorkerSwarm():
    jobs = numpy.zeros(20, dtype=bool)
    with WorkerSwarm(LOGGER) as swarm:
        for i in range(len(jobs)):
            swarm.add_task(get_task_fn(jobs, i))
    assert numpy.all(jobs)

def test_WorkerSwarm_nonparallel_allow_parallel_for_other_workers():
    jobs = []
    answer = numpy.arange(2)
    jobs2 = numpy.zeros(10, dtype=bool)
    with WorkerSwarm(LOGGER) as swarm:
        for i in range(len(answer)):
            swarm.add_task(get_append_fn(jobs, i, sleep=0.5), parallel=False)
        for i in range(len(jobs2)): # These should run quickly while the others are running one at a time.
            swarm.add_task(get_task_fn(jobs2, i), parallel=True)
    assert numpy.all(numpy.isclose(jobs, answer))
    assert numpy.all(jobs2)

if __name__ == '__main__':
    test_WorkerSwarm_nonparallel_allow_parallel_for_other_workers()
