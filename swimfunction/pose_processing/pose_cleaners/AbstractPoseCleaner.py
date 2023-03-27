import numpy

class AbstractPoseCleaner:
    ''' Abstract cleaner class
    '''
    def clean(self, poses: numpy.ndarray, *args, **kwargs):
        ''' Cleans an array of poses.
        '''
        raise NotImplementedError('Abstract method')
