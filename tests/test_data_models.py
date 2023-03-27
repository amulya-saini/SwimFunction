from swimfunction.data_models import PCAResult, Fish, SwimAssay, SimpleStack
from swimfunction.global_config.config import config
from swimfunction.pytest_utils import set_test_cache_access

import pytest

def __setup():
    ''' Allows fixture to be called directly if absolutely necessary.
    '''
    set_test_cache_access()

@pytest.fixture(autouse=True)
def SETUP_FILES():
    __setup()

def check_for_value_error(fn):
    got_it = False
    try:
        fn()
    except ValueError:
        got_it = True
    assert got_it, 'Should have thrown ValueError.'

def get_valid_swim_assay():
    vals = {'raw_coordinates': 1, 'likelihoods': 2, 'smoothed_coordinates': 3, 'smoothed_complete_angles': 4}
    return SwimAssay.SwimAssay(vals)

def get_valid_fish():
    f = Fish.Fish(name='M12')
    f[-1] = get_valid_swim_assay()
    return f

def test_PCAResult():
    PCAResult.PCAResult()
    PCAResult.PCAResult({'mean': None, 'pcs': None, 'norm_pcs': None, 'variances': None, 'positions': None, 'norm_positions': None})
    PCAResult.PCAResult(mean=None, pcs=None, norm_pcs=None, variances=None, positions=None, norm_positions=None)
    PCAResult.PCAResult(('mean', 'pcs', 'norm_pcs', 'variances', 'positions', 'norm_positions'))
    PCAResult.PCAResult('mean', 'pcs', 'norm_pcs', 'variances', 'positions', 'norm_positions')
    p = PCAResult.PCAResult('first','second','third','fourth','fifth','sixth')
    assert p.mean == 'first' and p['mean'] == 'first' and p[0] == 'first', 'PCAResult failed'
    assert p.variances == 'fourth' and p['variances'] == 'fourth' and p[3] == 'fourth', 'PCAResult failed'

def test_SwimAssay():
    get_valid_swim_assay()

def test_Fish():
    check_for_value_error(lambda: Fish.Fish())
    f = get_valid_fish()
    assert isinstance(f, (Fish.Fish)) and isinstance(f[-1], (SwimAssay.SwimAssay))
    f_as_dict = f.as_dict()
    assert not isinstance(f_as_dict, (Fish.Fish)) and not isinstance(f_as_dict[-1], (SwimAssay.SwimAssay))
    assert isinstance(f_as_dict, (dict)) and isinstance(f_as_dict[-1], (dict))

def test_simple_stack():
    capacity = 5
    ss = SimpleStack.SimpleStack(capacity)
    # Test get None from an empty SimpleStack
    assert ss.get() is None, 'Get None'
    # Test fill it up and get everything back
    for i in range(capacity):
        ss.put(i)
    assert ss.get() is not None, 'Get a non-None value'
    for x in reversed(range(capacity-1)):
        assert x == ss.get(), 'Get the rest non-None values.'
    assert ss.get() is None, 'Get None.'
    # Test one overwriting
    for i in range(capacity+1):
        ss.put(i)
    for x in reversed(range(1, capacity+1)):
        assert x == ss.get(), 'Get the all values.'
    # Test a bunch of overwriting
    vals = list('This is a really long sentence that will be split.')
    ss = SimpleStack.SimpleStack(capacity)
    for x in vals:
        ss.put(x)
    for x in reversed(vals[-capacity:]):
        assert x == ss.get(), 'Get the last inputs in reverse order.'

if __name__ == '__main__':
    __setup()
    test_PCAResult()
    # By default, load CSV to guarantee it runs all tests successfully and quickly
    # test_pca_calculation()
    # test_pca_calculation_with_pose_filtering()

