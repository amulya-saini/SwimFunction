''' Context manager for cached files.
'''
from os.path import getmtime
import threading
import warnings

import joblib
import numpy
import pathlib
import shutil
import time
import gc

import pandas

HDFKEY = 'df_with_missing'

def as_absolute_path(location) -> pathlib.Path:
    ''' Get the location as a Path expanded and resolved.
    '''
    return pathlib.Path(location).expanduser().resolve()

def get_modified_time(path):
    if pathlib.Path(path).exists():
        return getmtime(path)
    return None

class CACHE_MEMORY_FILE_DATA:
    def __init__(self, path, contents):
        self.path = path
        self.mtime = get_modified_time(path)
        self.contents = contents
        self.returned_time = time.time()
    def matches(self, path):
        make_uniform_posix = lambda p: as_absolute_path(p).as_posix()
        p1 = make_uniform_posix(self.path)
        p2 = make_uniform_posix(path)
        does_match = (p1 == p2) \
            and (get_modified_time(p1) == get_modified_time(p2))
        if does_match:
            self.returned_time = time.time()
        return does_match

class LimitedCacheMemory:
    def __init__(self, size=10):
        self.memory_size = size
        self.memory = []
    def memory_decorator(self, read_fn):
        def inner(fpath, verbose=False):
            fpath = pathlib.Path(fpath)
            for fmem in self.memory:
                if fmem.matches(fpath):
                    if verbose:
                        print(f'Returning cached {fpath.name}')
                    return fmem.contents
            contents = read_fn(fpath, verbose)
            if contents is not None:
                self.memory.append(CACHE_MEMORY_FILE_DATA(fpath, contents))
            if len(self.memory) > self.memory_size:
                self.memory = sorted(self.memory, key=lambda a: a.returned_time) # Sort oldest to newest
                self.memory.pop(0) # Pop the oldest memory chunk.
                gc.collect()
            return contents
        return inner

CACHE_MEMORY_SIZE = 10
limited_cache_memory = LimitedCacheMemory(CACHE_MEMORY_SIZE)

def read_numpy_cache(fpath, verbose=False):
    fpath = pathlib.Path(fpath)
    if not fpath.suffix == '.npy':
        raise ValueError('Numpy cache must use extension .npy')
    if not fpath.exists():
        return None
    start = time.time()
    rv = numpy.load(fpath)
    if verbose:
        print(f'Finished loading in {time.time() - start:.2f} seconds')
    return rv

def dump_numpy_cache(np_arr, fpath, verbose=False):
    fpath = pathlib.Path(fpath)
    if not fpath.suffix == '.npy':
        raise ValueError('Numpy cache must use extension .npy')
    start = time.time()
    numpy.save(fpath, np_arr)
    if verbose:
        print(f'Finished saving in {time.time() - start:.2f} seconds')

#@limited_cache_memory.memory_decorator
def read_cache(fpath, verbose=False):
    ''' Infer load method using extension.
        If extension is .h5, loads with pandas
        By default uses joblib.
    '''
    # from swimfunction.main_scripts.metric_tca import TensorIndex
    fpath = as_absolute_path(fpath)
    start = time.time()
    _rv =  None
    if not fpath.exists():
        _rv = None
    elif fpath.suffix == '.txt':
        with open(fpath, 'rt') as fh:
            _rv = fh.read()
    elif fpath.suffix == '.npy':
        _rv =  read_numpy_cache(fpath)
    elif fpath.suffix == '.h5':
        store = pandas.HDFStore(fpath.as_posix(), mode='r')
        _rv =  pandas.read_hdf(store, key=HDFKEY)
        store.close()
    else:
        _rv = joblib.load(fpath)
    if verbose:
        print(f'Finished loading in {time.time() - start:.2f} seconds')
    return _rv

def dump_cache(obj, fpath, verbose=False):
    ''' By default uses joblib, otherwise tries to infer save method using extension.
    Extensions:
        .txt (saves with open(fpath, 'wt').write(obj))
        .npy (saves using numpy)
        .h5 (saves assuming obj is pandas dataframe)
        [default] joblib.dump
    '''
    fpath = as_absolute_path(fpath)
    start = time.time()
    if obj is None:
        pass
    elif fpath.suffix == '.txt':
        with open(fpath, 'wt') as fh:
            fh.write(obj)
    elif fpath.suffix == '.npy':
        dump_numpy_cache(obj, fpath)
    elif fpath.suffix == '.h5':
        obj.to_hdf(fpath, key=HDFKEY, format='table', mode='w')
    else:
        # Dump to a tmp first, so it doesn't destroy the cache if the user exits the script in the middle of saving the file.
        tmp = fpath.as_posix() + '.partial'
        # Previously I used compress=3, protocol=4, but it was very slow and only reduced 100 to 60 memory units.
        # I'm accepting a storage increase for a significant speed increase.
        joblib.dump(obj, tmp, compress=0)
        shutil.move(tmp, fpath)
    if verbose:
        print(f'Finished saving in {time.time() - start:.2f} seconds')

class _Cache:
    __slots__ = ['_load_fn', '_contents', '_save']
    def __init__(self, load_fn):
        self._load_fn = load_fn
        self._contents = None
        self._save = False
    def getContents(self, default_val=None):
        ''' Get the contents of the cache.
        '''
        try:
            self._contents = self._load_fn()
        except Exception as e:
            print('Could not load contents of cache:', e)
        if self._contents is None:
            return default_val
        return self._contents
    def saveContents(self, contents):
        ''' Use this function to set the contents
        so that it saves properly when CacheContext exits.
        '''
        self._contents = contents
        self._save = True

class CacheContext:
    ''' Context Manager
    Handles cache i/o so you don't need to worry about its implementation.
    '''

    IO_LOCK = threading.Lock()
    __slots__ = ['cache_path', 'cache']

    def __init__(self, cache_path):
        '''
        Parameters
        ----------
        cache_path : str or pathlib.Path
            Path to the output file.
            If the output file already exists, the lock will not be acquired.
        '''
        self.cache_path = as_absolute_path(cache_path)
        with self.IO_LOCK:
            try:
                self.cache = _Cache(lambda: read_cache(self.cache_path))
            except Exception as e:
                warnings.warn(f'Cache could not be read: {self.cache_path.as_posix()} due to {e}')

    def __enter__(self):
        return self.cache

    def __exit__(self, type, value, traceback):
        if self.cache._save:
            with self.IO_LOCK:
                try:
                    dump_cache(self.cache._contents, self.cache_path)
                except Exception as e:
                    warnings.warn(f'Cache contents could not be written: {self.cache_path.as_posix()} due to {e}')
