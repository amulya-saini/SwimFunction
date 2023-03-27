import os
import pathlib

class FileLock:
    ''' Context Manager
    Pseudo-semaphore strategy for making a script "instance-safe".
    Using this function, you can safely run multiple instances of a script on a large number of files,
    where each instance acts on one file at a time.
    It looks first for the expected output file, then for a lock file,
    and finally makes a lock file if neither exist.

    Check whether lock value is true before continuing. See example below.

    Example:
    with FileLock('~/', 'newfile', 'newfile.png') as fl,
        if fl:
            save_figure(figure_data, 'newfile.png')
    '''
    def __init__(self, lock_dir, lock_name, output_path=None):
        '''
        Parameters
        ----------
        lock_dir : str or pathlib.Path
            location to save the lockfile.
            Could be inside /tmp or inside one of the working directories.
        lock_name : str
            Unique name of the lock, if the lock alread exists then the lock will not be acquired.
            I recommend using pathlib.Path().name of the input or output file.
        output_path : str or pathlib.Path, optional
            Path to the output file.
            If the output file already exists, the lock will not be acquired.
        '''
        self.lock_dir = pathlib.Path(lock_dir)
        self.lock_name = lock_name
        self.output_path = output_path
        self.lockfile = self.lock_dir / f'LOCK_{self.lock_name}.txt'
        self.lock_acquired = False
    def __enter__(self):
        self.lock_acquired = True
        if self.output_path is not None and pathlib.Path(self.output_path).exists():
            self.lock_acquired = False # Output already exists.
        if self.lockfile.exists():
            self.lock_acquired = False # Output is already being processed (lockfile exists)
        if self.lock_acquired:
            with open(self.lockfile.as_posix(), 'wt') as fh:
                fh.write('LOCKED')
        return self.lock_acquired
    def __exit__(self, type, value, traceback):
        if self.lock_acquired:
            os.remove(self.lockfile.as_posix())
