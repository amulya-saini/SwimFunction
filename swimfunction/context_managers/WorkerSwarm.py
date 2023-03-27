''' Use a worker swarm to complete independent, parallel tasks.
    Add tasks (functions) to perform, then wait for them to complete.
    Please use WorkerSwarm as context manager. It also protects against
    nested parallelism. Priority is allowed for the task priority queue,
    by default it's handled according to swarm nesting to try to optimize performance.

   _WorkerSwarm does not act as a context manager,
    nor does it protect against nested parallelism.
    await_tasks_complete is context dependent
    Nested contexts is allowed.
'''
from queue import PriorityQueue, Queue, Empty

import traceback
import logging
import threading
import uuid

class _WorkerSwarm:

    def __init__(self, num_workers: int, logger: logging.Logger=None):
        ''' DO NOT CALL INIT DIRECTLY! USE INITIALIZE_SWARM
        AND GET_SWARM TO AVOID NESTED PARALLELISM.
        '''
        self.verbose = False
        self.ended = False
        self.task_var_lock = threading.Lock() # Protects task_dict
        self.task_run_lock = threading.Semaphore(0) # Counts tasks remaining
        self.task_await_lock = threading.Semaphore(0) # Counts tasks completed
        self.tasks = PriorityQueue()
        self.sequential_tasks = Queue()
        self.sequential_task_lock = threading.Lock() # Ensures one sequential task at a time.
        self.task_dict = {}
        self.task_await_set = set()
        self.keep_going = True
        if logger is None:
            from swimfunction import loggers
            logger = loggers.get_console_logger(__name__, level=logging.INFO)
        self.logger = logger
        self.workers = [
            threading.Thread(target=self._run_worker, args=(i,))
            for i in range(num_workers)]
        list(map(lambda w: w.start(), self.workers))

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def add_task(self, fn, priority: int=1, parallel: bool=True) -> int:
        '''
        Parameters
        ----------
        fn
            Function to be run as a task
        priority: int, None
            Leave this as None to get default behavior.
            If you are not nesting Swarms, then you can use priority safely.
            Otherwise, I do not guarantee there won't be inefficiency.
        parallel: bool, default=True
            If true, this task can be processed parallel with any others.
            All jobs with parallel=False must be processed one at a time,
            though prioritized before (and parallel with) parallel=True tasks,
            and priority between parallel=False jobs will be ignored.
        '''
        if self.ended:
            raise RuntimeError('Worker swarm already ended. Sorry, cannot add task!')
        uid = uuid.uuid4().int
        with self.task_var_lock:
            self.task_dict[uid] = fn
            if parallel:
                self.tasks.put((priority, uid))
            else:
                self.sequential_tasks.put((priority, uid))
            self.task_await_set.add(uid)
        self.task_run_lock.release() # Indicate task is available
        return uid

    def _run_worker(self, worker_id):
        if self.ended:
            raise RuntimeError('Worker swarm already ended. Sorry, cannot run!')
        self.logger.debug('Worker %s started.', worker_id)
        break_away = False
        while self.keep_going or self.task_dict:
            self.task_run_lock.acquire() # Wait for task to be available.
            task_id = None
            priority = None
            try:
                task_fn, task_id, priority = self._pop_sequential_task()
                if task_id is None and task_fn is not None:
                    print('HEY!')
                if task_id is None:
                    task_fn, task_id, priority = self._pop_task()
                if task_id is None and task_fn is not None:
                    print('HEY!')
                if task_id is not None:
                    self._indicate_task_running(worker_id, task_id, priority)
                    task_fn()
            except Exception as e:
                # Log everything that isn't about the queue being empty
                if not isinstance(e, Empty):
                    self.logger.warning('Error caught: %s', traceback.format_exc())
            finally:
                # Avoid deadlock.
                if task_id is not None:
                    self._indicate_task_complete(worker_id, task_id, priority)
                else:
                    # Indicate task is still available (nothing was run)
                    self.task_run_lock.release()
                    # Nothing parallel could be run, only sequential tasks remain.
                    break_away = True
            if break_away:
                break
        self.logger.debug('Worker %s ended.', worker_id)

    def _pop_sequential_task(self):
        fn, task_id, priority = None, None, None
        # Allow one sequential task to be run at a time.
        if self.sequential_task_lock.acquire(timeout=0.1):
            try:
                priority, task_id = self.sequential_tasks.get(timeout=0.1)
                with self.task_var_lock:
                    inner_fn = self.task_dict.pop(task_id)
                    fn = self._sequential_fn_task(inner_fn)
            except Empty:
                self.sequential_task_lock.release()
        return fn, task_id, priority

    def _sequential_fn_task(self, fn):
        def task():
            res = fn()
            self.sequential_task_lock.release()
            return res
        return task

    def _pop_task(self):
        fn, task_id, priority = None, None, None
        priority, task_id = self.tasks.get(timeout=0.1)
        with self.task_var_lock:
            fn = self.task_dict.pop(task_id)
        return fn, task_id, priority

    def _indicate_task_running(self, worker_id, task_id, _priority):
        if self.verbose:
            self.logger.debug('Worker %d running task %d', worker_id, task_id)

    def _indicate_task_complete(self, worker_id, task_id, _priority):
        if self.verbose:
            self.logger.debug('Worker %d task %d complete.', worker_id, task_id)
        self.task_await_lock.release()

    def await_all_tasks_complete(self):
        ''' Wait for all tasks to complete
        '''
        if self.ended:
            return
        try:
            while self.task_await_set:
                self.task_await_set.pop()
                self.task_await_lock.acquire()
        except KeyError as _e:
            pass # If set is empty, no problem, we just made a minor mistake.

    def end(self):
        ''' End the swarm after all tasks are complete.
        '''
        self.logger.debug('ENDING SWARM')
        self.keep_going = False
        self.await_all_tasks_complete()
        for _ in self.workers:
            self.task_run_lock.release() # Allow all workers to complete.
        list(map(lambda w: w.join(), self.workers))
        self.ended = True

class WorkerSwarm:
    ''' Handles specific set of tasks given to the swarm.
    Use as a context manager, it will wait for tasks to complete.
    You can nest these inside of each other,
    though inner nestings all get the same priority.
    The outermost nested context will end the swarm.
    '''
    outer_priority = 2 # Default for single or outer swarm
    inner_priority = 1 # For internal, nested swarm

    num_allowed_workers = 1

    DISABLED = False

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        if WorkerSwarm.DISABLED:
            return
        self.await_tasks_complete()
        # If it's the outermost swarm, end it.
        if self.priority == WorkerSwarm.outer_priority:
            self.logger.debug('OUTER SWARM EXITED')
            WorkerSwarm._ACTIVE_SWARM.end()
            WorkerSwarm._ACTIVE_SWARM = None
        else:
            self.logger.debug('INNER SWARM EXITED')

    _ACTIVE_SWARM = None

    def __init__(self, logger: logging.Logger=None, verbose=False):
        from swimfunction import loggers
        self.logger = logger if logger is not None else loggers.get_console_logger(__name__)
        self.verbose = verbose
        self.task_await_set = set()
        self.task_await_lock = threading.Semaphore(0) # Counts completed tasks
        self.logger = logger if logger is not None else loggers.get_console_logger(__name__)
        if WorkerSwarm._ACTIVE_SWARM is None or WorkerSwarm._ACTIVE_SWARM.ended:
            WorkerSwarm._ACTIVE_SWARM = _WorkerSwarm(WorkerSwarm.num_allowed_workers, logger)
            self.priority = WorkerSwarm.outer_priority
        else:
            self.priority = WorkerSwarm.inner_priority
        WorkerSwarm._ACTIVE_SWARM.verbose = verbose

    def add_task(self, fn, priority: int=None, parallel: bool=True):
        '''
        Parameters
        ----------
        fn
            Function to be run as a task
        priority: int, None
            Leave this as None to get default behavior.
            If you are not nesting Swarms, then you can use priority safely.
            Otherwise, I do not guarantee there won't be inefficiency.
        parallel: bool, default=True
            If true, this task can be processed parallel with any others.
            All jobs with parallel=False must be processed one at a time,
            though prioritized before (and parallel with) parallel=True tasks,
            and priority between parallel=False jobs will be ignored.
        '''
        if WorkerSwarm.DISABLED:
            return fn()
        def task_fn(outer_self):
            res = fn()
            outer_self.task_await_lock.release()
            return res
        self.task_await_set.add(
            WorkerSwarm._ACTIVE_SWARM.add_task(
                lambda outer_self=self: task_fn(outer_self),
                self.priority if priority is None else priority,
                parallel=parallel)
        )

    def await_tasks_complete(self):
        if WorkerSwarm.DISABLED:
            return
        try:
            while self.task_await_set:
                self.task_await_set.pop()
                self.task_await_lock.acquire()
        except KeyError as _e:
            pass # If set is empty, no problem, we just made a minor mistake.
