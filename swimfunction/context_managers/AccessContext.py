from swimfunction.global_config.config import config, CacheAccessParams
import pathlib

class AccessContext:
    ''' Context Manager
    Temporarily changes access params
    (cache root and experiment name, whichever are not None in __init__)
    '''
    __slots__ = ['original_access', 'temporary_access']
    def __init__(self, cache_root=None, experiment_name=None):
        '''
        Parameters
        ----------
        cache_path : str or pathlib.Path
            Path to the output file.
            If the output file already exists, the lock will not be acquired.
        '''
        if cache_root is not None and not pathlib.Path(cache_root).exists():
            raise RuntimeError('That cache root does not exist. For your safety, the program must exit now.')
        self.original_access = config.access_params
        self.temporary_access = CacheAccessParams(
            cache_root if cache_root is not None else config.access_params.cache_root,
            experiment_name if experiment_name is not None else config.experiment_name,
        )
    def __enter__(self):
        config.set_access_params(self.temporary_access)
    def __exit__(self, type, value, traceback):
        config.set_access_params(self.original_access)

    @staticmethod
    def available_experiments_with_videos(cache_root: pathlib.Path=None):
        ''' Find all experiments that have a folder called "videos"
        '''
        if cache_root is None:
            cache_root = config.access_params.cache_root
        return [p.parent.name for p in pathlib.Path(cache_root).glob('*/videos')]
