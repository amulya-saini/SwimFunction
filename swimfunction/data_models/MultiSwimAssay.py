from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.SwimAssay import SwimAssay
import gc

class MultiSwimAssay(list):
    ''' Similar to a Fish object, but instead of holding multiple assays for the same fish,
        it holds swim assay data for multiple fish in the same video.
    Has helper functions to load and save itself using fish_manager
    '''
    __slots__ = ['basename', 'condition']

    def __init__(self, basename, condition):
        '''
        Parameters
        ----------
        basename : str
            the name of the assay (is the basename of the cache file)
        condition : str or int
            could be weeks post-injury, genotype, or disease condition.
        '''
        self.basename = basename
        self.condition = condition

    def as_dict(self):
        ''' The object as a dictionary, calling as_dict on all values.
        Remember, values should all be SwimAssay objects.
        '''
        return {
            'basename': self.basename,
            'condition': self.condition,
            'swim_data': [swim_assay.as_dict() for swim_assay in self]
        }

    def load(self):
        ''' Clears self dictionary, then loads a dictionary of dictionaries
            using a fish_manager object, converts dictionaries into
            SwimAssay objects and stores them in self.swims_dict

            Returns
            -------
            self : MultiSwimAssay
                for convenient function chaining

            Raises
            ------
            IOError
                if tried to access an incorrect cache file.
        '''
        assay_dict = FDM.load_fish_data(self.basename)
        # Check successful load
        assert self.basename == assay_dict['basename'], \
            f'Loaded incorrect basename: {assay_dict["basename"]} != {self.basename}'
        assert self.condition == assay_dict['condition'], \
            f'Loaded incorrect condition: {assay_dict["condition"]} != {self.condition}'
        for d in assay_dict['swim_data']:
            self.append(SwimAssay(d))
        return self

    def save(self):
        ''' Converts SwimAssays into tuples and saves a dictionary of tuples using a fish_manager object.
        '''
        FDM.save_fish_data(self.basename, self.as_dict())

    def swim_keys(self):
        ''' Returns list of keys to swim assays.
        (Maintains compatability with the Fish object,
        though here the keys are indexes to other fish in the same assay,
        not weeks post-injury.)
        '''
        return list(range(len(self)))

    def __str__(self):
        return self.basename

    def delete(self):
        del self
        gc.collect()

