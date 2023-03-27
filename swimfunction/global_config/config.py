''' Config required to get global variables
and cache access parameters.
'''
import pathlib
from collections import namedtuple
from configparser import ConfigParser
import matplotlib.patches as mpatches


class CacheAccessParams:
    ''' Basic way to locate an experiment cache.
    Cache root is the location of the experiment folders,
    experiment name is the folder of the specific experiment.
    '''
    __slots__ = ['cache_root', 'experiment_name']
    def __init__(self, cache_root, experiment_name):
        self.cache_root = pathlib.Path(cache_root).expanduser().resolve()\
            if cache_root is not None else cache_root
        self.experiment_name = experiment_name

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, CacheAccessParams) \
                and __o.cache_root == self.cache_root \
                and __o.experiment_name == self.experiment_name:
            return True
        return False

    def __str__(self):
        return f'root: {self.cache_root}\nexperiment: {self.experiment_name}'

    @staticmethod
    def get_test_access():
        ''' For unit and module testing
        '''
        from swimfunction.FileLocations import get_test_root
        return CacheAccessParams(get_test_root(), 'test_experiment')

class _FishConfigParserSingleton(ConfigParser):
    ''' If you ever make new config files, this is where you need to update the name.
    '''
    _config_fname = 'config.ini'
    config_path = pathlib.Path(__file__).expanduser().resolve().parent / _config_fname

    @staticmethod
    def split_string_list(list_string):
        return list_string.rstrip(',').lstrip('[').rstrip(']').rstrip(',').split(',')

    @staticmethod
    def to_string_list(list_string):
        # Remove all single and double quotes
        return [
            x.replace('\'','').replace('\"','')
            for x in _FishConfigParserSingleton.split_string_list(list_string)]

    @staticmethod
    def to_int_list(list_string):
        return [int(x) for x in _FishConfigParserSingleton.split_string_list(list_string)]

    @staticmethod
    def to_float_list(list_string):
        return [float(x) for x in _FishConfigParserSingleton.split_string_list(list_string)]

    @staticmethod
    def to_bool_list(list_string):
        def to_bool(s):
            if s == 'true':
                return True
            if s == 'false':
                return False
            else:
                raise RuntimeError('Config bool list could not be parsed correctly.')
        return [to_bool(x) for x in _FishConfigParserSingleton.split_string_list(list_string)]

    @staticmethod
    def parse_list(list_string, list_type_conversion_fn):
        ''' Parses 1D or 2D lists.
        '''
        list_string = list_string.replace(' ','').replace('\n','')
        if '],[' in list_string:
            list_string = [
                list_type_conversion_fn(sublist_string)
                for sublist_string in list_string.split('],[')]
        else:
            list_string = list_type_conversion_fn(list_string)
        return list_string

    def __init__(self):
        self.getpath = lambda _section, _key: None
        super().__init__(converters={
            'path': pathlib.Path
        })
        self.DEFAULT_ACCESS_PARAMS = None
        self.access_params = None

    @property
    def experiment_name(self):
        ''' Get active experiment name
        '''
        return self.access_params.experiment_name

    @property
    def cache_root(self):
        ''' Get active cache root (for saving stuff and such)
        '''
        return self.access_params.cache_root

    def set_access_params(self, access_params: CacheAccessParams):
        self.access_params = access_params

    def getintlist(self, *args, **kwargs):
        return self.parse_list(self.get(*args, **kwargs), self.to_int_list)

    def getfloatlist(self, *args, **kwargs):
        return self.parse_list(self.get(*args, **kwargs), self.to_float_list)

    def getstringlist(self, *args, **kwargs):
        return self.parse_list(self.get(*args, **kwargs), self.to_string_list)

    def getboollist(self, *args, **kwargs):
        return self.parse_list(self.get(*args, **kwargs), self.to_bool_list)

    def getnamedtuple(self, labelsection, labelkey, valuessection, valueskey):
        ''' Returns namedtuple where both keys and values are strings
        example: BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
        '''
        return namedtuple(f'{labelkey}Tuple', self.getstringlist(labelsection, labelkey))\
                (*self.getstringlist(valuessection, valueskey))

    def getstringdict(self, labelsection, labelkey, valuessection, valueskey):
        '''
        example: BEHAVIORS = config.getstringdict('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
        '''
        return dict(zip(
            self.getstringlist(labelsection, labelkey),
            self.getstringlist(valuessection, valueskey)))

    def getbooldict(self, labelsection, labelkey, valuessection, valueskey):
        '''
        example: BEHAVIORS = config.getstringdict('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
        '''
        return dict(zip(
            self.getstringlist(labelsection, labelkey),
            self.getboollist(valuessection, valueskey)))

    def getfloatdict(self, labelsection, labelkey, valuessection, valueskey):
        '''
        example: BEHAVIOR_COLORS_DICT = config.getfloatdict(
            'BEHAVIORS', 'names', 'BEHAVIORS', 'colors')
        '''
        return dict(zip(
            self.getstringlist(labelsection, labelkey),
            self.getfloatlist(valuessection, valueskey)))

    def getmatplotlibpatches(self, labelsection, labelkey, colorssection, colorskey):
        '''
        example: behavior_legend_handles = config.getmatplotlibpatches(
            'BEHAVIORS', 'names', 'BEHAVIORS', 'colors')
        '''
        return [
            mpatches.Patch(color=c, label=l)
            for l, c in self.getfloatdict(labelsection, labelkey, colorssection, colorskey)
        ]

    def read(self, *args, **kwargs):
        super().read(*args, **kwargs)
        self.DEFAULT_ACCESS_PARAMS = CacheAccessParams(
            cache_root=self.getpath('FOLDERS', 'cache_root'),
            experiment_name=self.get('EXPERIMENT DETAILS', 'experiment_name'))
        self.set_access_params(self.DEFAULT_ACCESS_PARAMS)

config = _FishConfigParserSingleton()

config.read(config.config_path.as_posix())

# ------- Examples -------
# from swimfunction.global_config.config import config
# FRAMES_PER_SECOND = config.getint('VIDEO', 'fps')
# BEHAVIORS = config.getnamedtuple('BEHAVIORS', 'names', 'BEHAVIORS', 'symbols')
# BEHAVIOR_COLORS = config.getfloatlist('BEHAVIORS', 'colors')
# BEHAVIOR_COLORS_DICT = config.getfloatdict('BEHAVIORS', 'symbols', 'BEHAVIORS', 'colors')
# NAME_TO_COLOR = config.getfloatdict('BEHAVIORS', 'names', 'BEHAVIORS', 'colors')
# PPW = config.getint('BEHAVIOR_ANNOTATION', 'poses_per_window')
# POINTS_PER_POSE = config.getint('POSES', 'points_per_pose')
# WPI_OPTIONS = config.getintlist('EXPERIMENT DETAILS', 'assay_labels')
# UNINJURED_WEEK = config.getint('EXPERIMENT DETAILS', 'uninjured_assay_label')
# ANGLE_CHANGE_Z_LIM = config.getint('POSE FILTERING', 'angle_change_z_lim')
# FRAME_SIDE_BUFFER_WIDTH = config.getint('POSE FILTERING', 'frame_side_buffer_width')
# FRAME_REAR_BUFFER_WIDTH = config.getint('POSE FILTERING', 'frame_rear_buffer_width')
# MIN_LIKELIHOOD_THRESHOLD = config.getfloat('POSE FILTERING', 'min_likelihood_threshold')
