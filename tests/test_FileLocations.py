import pathlib

from swimfunction.global_config.config import config, CacheAccessParams
from swimfunction.pytest_utils import set_test_cache_access
from swimfunction import FileLocations

config.set('TEST', 'test', 'true')

def test_set_config():
    roots = ['/tmp/dsa','/tmp/j5ter','/tmp/fyn','/tmp/cW','/tmp/zwef']
    names = ['fdhgf','ghjl','twedr','fklgh','bvcx']
    for cr, en in zip(roots, names):
        config.set_access_params(CacheAccessParams(cr, en))
        assert FileLocations.get_cache_root().name == pathlib.Path(cr).name
        assert FileLocations.get_experiment_name() == en
    set_test_cache_access()
    ta = config.access_params
    assert FileLocations.get_cache_root().name == ta.cache_root.name
    assert FileLocations.get_experiment_name() == ta.experiment_name

if __name__ == '__main__':
    test_set_config()
