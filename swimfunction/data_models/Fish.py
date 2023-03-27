import traceback
from swimfunction.data_access.fish_manager import DataManager as FDM
from swimfunction.data_models.SwimAssay import SwimAssay
import gc

class Fish(dict):
    ''' Fish extends Dict to include the name of the fish.

    General structure:
    Key: number (weeks-post-injury)
    Value: SwimAssay object

    Also contains its own name.

    Has helper functions to load and save itself using fish_manager
    '''
    def __init__(self, *args, **kwargs):
        ''' Must initialize with a name.

        Raises
        ------
        ValueError
            If "name" is not provided.
        '''
        if len(kwargs) == 0 and len(args) == 1 and isinstance(args[0], str):
            name = args[0]
            args = []
            kwargs = dict(name=name)
        elif 'name' not in kwargs:
            raise ValueError('You must give Fish a name. Usage example: Fish(name=\'F3\')')
        super().__init__(*args, **kwargs)
        self.name = self['name']

    def as_dict(self):
        ''' Returns a dictionary with self's contents, calling as_dict on all values.
        Remember, values should all be SwimAssay objects.
        '''
        return {
            k: v.as_dict()
               if isinstance(v, (SwimAssay)) else v
            for k, v in self.items()
        }

    def load(self):
        ''' Clears self dictionary, then loads a dictionary of dictionaries
            using a fish data manager object, converts dictionaries into
            SwimAssay objects and stores them in self

            Returns
            -------
            self
                for convenient function chaining

            Raises
            ------
            IOError
                If tried to access an incorrect cache file.
        '''
        fish_dict = FDM.load_fish_data(self.name)
        if fish_dict is None:
            fish_dict = {}
        self.clear()
        for k, v in fish_dict.items():
            if isinstance(v, (dict)):
                self[k] = SwimAssay(v, fish_name=self.name, assay_label=k)
            else:
                self[k] = v
        if 'name' in self and self.name != self['name']:
            raise IOError(f'You cannot load cache for fish named \
                {self["name"]} into a Fish object named {self.name}.')
        return self

    def save(self):
        ''' Converts SwimAssays into tuples and
        saves a dictionary of tuples using a fish data manager object.
        '''
        FDM.save_fish_data(self.name, self.as_dict())

    def add_caudal_fin_coordinates(self):
        ''' Adds caudal fin coordinates and likelihoods
        to each swim assay, then saves the fish data.
        '''
        if len(self.swim_keys()) == 0:
            print('Will not load caudal fins for an empty fish object. Did you forget to call fish.load() ?')
            return
        for assay_label in self.swim_keys():
            try:
                fin_coords, fin_likelihoods = FDM.read_caudal_fin_pose_file(self.name, assay_label)
                self[assay_label].caudal_fin_coordinates = fin_coords
                self[assay_label].caudal_fin_likelihoods = fin_likelihoods
            except Exception as e:
                print(traceback.format_exc)
                print(f'Cannot get caudal fin data for {assay_label} due to {e}')
        self.save()
        return self

    def swim_keys(self):
        ''' Returns sorted list of keys to swim assays
        '''
        return sorted([k for k in self.keys() if isinstance(self[k], (SwimAssay))])

    def __str__(self):
        return self.name

    def delete(self):
        ''' Clear the object from memory.
        '''
        del self
        gc.collect()
