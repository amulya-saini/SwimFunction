# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import sys
import time
import threading

_PREV_LEN = 0
_START = None
_TOTAL = 0
_IGNORED_N = 0

_RUNNING_COUNT_LOCK = threading.Lock()
_RUNNING_COUNT = 0

def init(total, start_i=0):
    global _START
    global _TOTAL
    global _IGNORED_N
    global _RUNNING_COUNT
    _START = time.time()
    _TOTAL = total
    _IGNORED_N = start_i
    _RUNNING_COUNT = start_i

def sec_to_hr_min_sec(seconds):
    h = seconds // (60*60)
    m = (seconds % (60*60)) // (60)
    s = seconds % 60
    return (h, m, s)

def _get_time_remaining(count):
    ellapsed = time.time() - _START
    completed = (count - _IGNORED_N) / (_TOTAL - _IGNORED_N)
    remaining = 1 - completed
    if completed != 0:
        sec_per_percent = ellapsed / completed
    else:
        sec_per_percent = 99999
    sec_remaining = int(remaining * sec_per_percent)
    return sec_remaining

def _append_space(msg):
    global _PREV_LEN
    original_len = len(msg)
    while len(msg) < _PREV_LEN:
        msg = msg + ' '
    _PREV_LEN = original_len
    return msg + '\r'

def progress(count, status='', total=None):
    if total is None:
        total = _TOTAL
    if total == 0:
        print('No tasks to report with Progress')
        return
    sec_remaining = _get_time_remaining(count)
    tr_hr, tr_min, tr_sec = sec_to_hr_min_sec(sec_remaining)
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    msg = '[%s] %s%s ...%s, ETA %d:%02d:%02d' % (bar, percents, '%', status, tr_hr, tr_min, tr_sec)
    sys.stdout.write(_append_space(msg))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)

def increment(status='', total=None):
    ''' Threadsafe alternative to progress(...)
    '''
    global _RUNNING_COUNT
    global _RUNNING_COUNT_LOCK
    with _RUNNING_COUNT_LOCK:
        try:
            _RUNNING_COUNT += 1
            progress(_RUNNING_COUNT, status, total)
        except:
            pass

def finish():
    bar_len = 60
    bar = '=' * bar_len
    tr_hr, tr_min, tr_sec = sec_to_hr_min_sec(time.time() - _START)
    msg = '[%s] %s, Finished %d tasks in %d:%02d:%02d' % (bar, '100%', _TOTAL, tr_hr, tr_min, tr_sec)
    sys.stdout.write(_append_space(msg))
    sys.stdout.write('\n')

class Progress:
    ''' Context manager for the module's progress functions.
    Calls init and finish on enter and exit, respectively.
    '''
    def __enter__(self):
        init(self.total, self.start_i)
        return self

    def __exit__(self, type, value, traceback):
        finish()

    def __init__(self, total: int, start_i: int=0):
        self.total = total
        self.start_i = start_i

    def progress(self, count, status=''):
        ''' Alias for progress.progress
        '''
        progress(count, status)

    def increment(self, status=''):
        ''' Alias for progress.increment
        '''
        increment(status)
