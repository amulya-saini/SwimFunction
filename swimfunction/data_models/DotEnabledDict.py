class DotEnabledDict(dict):
    ''' Overloads dictionary to allow dot operator access.
    Note: all attributes will be assigned as dictionary key-value pairs.
    If you don't want this functionality, don't use DotEnabledDict
    '''

    def as_dict(self):
        '''
        Returns
        -------
        dict
            the standard python dictionary version of self.
        '''
        return dict(self)

    def from_dict(self, d):
        ''' Clears then sets its keys/values from a dictionary.
        Normally you'd want to do this at initialization, but you can
        call this function if you want to load after creating the object.

        Returns
        -------
        self : DotEnabledDict
            (for convenient function chaining)
        '''
        self.clear()
        if d is None:
            return self
        for k, v in d.items():
            self[k] = v
        return self

    ''' Overload dictionary attributes to allow dot-operated access
    '''
    def __getattr__(self, attr):
        if attr not in self:
            raise KeyError(attr)
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]
