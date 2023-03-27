''' Dynamic Time Warping helper function
'''
import dtw
import numpy

def dynamic_time_warp(query, template, **kwargs):
    ''' Ensures lengths are equal.
    Selects the warping that has the lowest distance.
    '''
    if query.shape[0] > template.shape[0]:
        _template = numpy.zeros_like(query)
        _template[:template.shape[0], ...] = template
        template = _template
    elif query.shape[0] < template.shape[0]:
        _query = numpy.zeros_like(template)
        _query[:query.shape[0], ...] = query
        query = _query
    w1 = dtw.dtw(query, template, **kwargs)
    w2 = dtw.dtw(query, template * -1, **kwargs)
    return w1 if w1.distance < w2.distance else w2
