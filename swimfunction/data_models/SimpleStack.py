class SimpleStack:
    ''' Super simple wrap-around LIFO queue with maximum size.
    The standard library LifoQueue does not wrap around given a max size.
    '''
    __slots__ = ['maxsize', '_contents', '_i']
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._contents = [None for _ in range(maxsize)]
        self._i = 0

    def _prev_i(self):
        return (self._i - 1) % self.maxsize

    def _next_i(self):
        return (self._i + 1) % self.maxsize

    def put(self, x):
        ''' Adds value. Gets rid of bottom value if already full.
        '''
        self._contents[self._next_i()] = x
        self._i = self._next_i()

    def get(self):
        ''' Removes and returns top value. May be None.
        '''
        rv = self._contents[self._i]
        self._contents[self._i] = None
        self._i = self._prev_i()
        return rv
