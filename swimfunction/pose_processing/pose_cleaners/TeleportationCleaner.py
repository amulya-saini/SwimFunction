''' Cleaners clean data.
'''
import functools
import numpy

from swimfunction.pose_processing.pose_cleaners.AbstractPoseCleaner import AbstractPoseCleaner

class TeleportationCleaner(AbstractPoseCleaner):
    ''' Ensures that two adjacent datapoints are not too far apart.
    Assumes positions are in euclidean space.

    Basic process:
        Goes through positions looking for places
            where the position changes more than is acceptable.
        Keeps the last acceptable point in that case.
        Special case: if the next three differences are valid,
            consider it a good point and assume a mistake occurred.

    Pseudocode:
        Define maximum allowable distance (MAD) = 45
        Define minimum frames required to overturn rejection (MOR) = 4
        For each keypoint on the fish
            For each frame with change > MAD
                Add frame and its next MOR frames to QUESTIONABLES
            For each frame in QUESTIONABLES
                Ignore if
                    Either the next MOR frames have distances <= MAD
                    Or its distance from last valid point <= MAD
                Replace with last valid point
    '''
    def __init__(self, max_allowed_position_change, min_to_overturn=4, verbose=False):
        self.max_allowed_position_change = max_allowed_position_change
        # Minimum good transitions to overturn a rejection
        self.min_to_overturn = min_to_overturn
        self.verbose = verbose

    def report_bad_positions(self, arr, compare_fn):
        ''' Prints the number of large jumps in the data.
        '''
        if self.verbose:
            differences = compare_fn(arr[:-1], arr[1:])
            num_bad = len(differences[differences > self.max_allowed_position_change])
            if num_bad > 0:
                print(f'Has {num_bad} bad jumps')

    def get_set_of_questionable_indices(self, differences) -> set:
        ''' Gets indices for all potentially questionable jumps.
        Ignores those whose future min_to_overturn changes are all ok
        '''
        too_far_from_previous = set()
        def add_to_list(x_i):
            # Add to list unless the next min_to_overturn changes are not too extreme
            if (x_i > len(differences)-self.min_to_overturn \
                or not numpy.all(
                    differences[
                        x_i:x_i+self.min_to_overturn
                    ] <= self.max_allowed_position_change)):
                too_far_from_previous.add(x_i)
        # Add the point and their min_to_overturn future neighbors too
        for i in numpy.where(differences > self.max_allowed_position_change)[0] + 1:
            add_to_list(i)
            for j in range(i+1, min(i+self.min_to_overturn+1, len(differences)+1)):
                add_to_list(j)
        return too_far_from_previous

    def _clean1d2d(self, arr1d2d, compare_fn):
        ''' Goes through positions looking for places where
        the position changes more than is acceptable.
        Keeps the last acceptable point in that case.
        Special case: if the next min_to_overturn differences are valid,
        consider it a good point and assume a mistake occurred.

        See pseudocode in the class description.

        Returns
        -------
        tuple
            (cleaned_array, was_cleaned_mask) where mask is
            1 for cleaned poses, 0 for original pose.
        '''
        cleaned = numpy.asarray(arr1d2d).copy()
        mask = numpy.zeros(cleaned.shape[0], dtype=bool)
        self.report_bad_positions(cleaned, compare_fn)
        # assume begin in a valid position
        differences = compare_fn(cleaned[:-1], cleaned[1:])
        too_far_from_previous = self.get_set_of_questionable_indices(differences).union(
            numpy.where(numpy.isnan(arr1d2d))[0])
        # Since we'll be changing the original array, must work off a copy.
        counter = 0
        for i in sorted(list(too_far_from_previous)):
            last_valid_i = i
            while last_valid_i in too_far_from_previous:
                last_valid_i -= 1
            last_valid_p = cleaned[last_valid_i]
            not_nan = numpy.all(~numpy.isnan(arr1d2d[i]))
            prev_is_nan = numpy.any(numpy.isnan(arr1d2d[i-1]))
            # if this point is after a nan or close to the last valid point, keep it.
            if not_nan and (
                    compare_fn(cleaned[i], last_valid_p) <= self.max_allowed_position_change \
                    or prev_is_nan):
                too_far_from_previous.remove(i)
            else:
                # Otherwise, fill in with last valid position
                counter += 1
                cleaned[i] = last_valid_p
                mask[i] = 1
                if compare_fn(cleaned[i], cleaned[i-1]) > self.max_allowed_position_change:
                    counter -= 1
        self.report_bad_positions(cleaned, compare_fn)
        return cleaned, mask

    def _compare2d(self, a, b):
        rv = None
        if len(a.shape) == 2:
            rv = numpy.linalg.norm(a - b, axis=1)
        else:
            rv = numpy.linalg.norm(a - b)
        if isinstance(rv, numpy.ndarray):
            rv[numpy.where(numpy.isnan(rv))] = 0
        elif numpy.isnan(rv):
            rv = 0
        return rv
                
    def _compare1d(self, a, b):
        rv = numpy.abs(a - b)
        if isinstance(rv, numpy.ndarray):
            rv[numpy.where(numpy.isnan(rv))] = 0
        elif numpy.isnan(rv):
            rv = 0
        return rv

    def clean1d(self, arr1d: numpy.ndarray):
        '''
        Returns
        -------
        tuple
            (cleaned_array, was_cleaned_mask) where mask is
            1 for cleaned poses, 0 for original pose.
        '''
        return self._clean1d2d(arr1d, self._compare1d)

    def clean2d(self, arr2d: numpy.ndarray):
        '''
        Returns
        -------
        tuple
            (cleaned_array, was_cleaned_mask) where mask is
            1 for cleaned poses, 0 for original pose.
        '''
        return self._clean1d2d(
            arr2d,
            self._compare2d)

    def clean3d(self, arr3d: numpy.ndarray):
        '''
        Returns
        -------
        tuple
            (cleaned_array, was_cleaned_mask) where mask is
            1 for cleaned poses, 0 for original pose.
        '''
        cleaned = arr3d.copy()
        masks = numpy.empty((arr3d.shape[1], arr3d.shape[0]), dtype=bool)
        for i in range(arr3d.shape[1]):
            cleaned[:, i, :], masks[i, :] = self.clean2d(cleaned[:, i, :])
        return cleaned, functools.reduce(lambda a, b: a & b, masks)

    def clean(self, poses: numpy.ndarray):
        ''' Decides whether to clean in 1d, 2d, or 3d.

        IMPORTANT NOTE: Whatever you give to this function
        will be cleaned as if it is one experiment.
        You must never give multiple experiments to this cleaner.

        Returns
        -------
        tuple
            (cleaned_array, was_cleaned_mask)
            where mask is 1 for cleaned poses, 0 for original pose.
        '''
        poses = numpy.asarray(poses)
        shape = poses.shape
        ndim = len(shape)
        assert ndim == 1 or (ndim <= 3 and shape[-1] == 2), \
            f'Got {ndim} dimensions {shape}. You must provide \
            a list of coordinates with shape () (, 2) (, , 2)'
        if ndim == 1:
            return self.clean1d(poses)
        if ndim == 2:
            return self.clean2d(poses)
        if ndim == 3:
            return self.clean3d(poses)
        raise ValueError(f'Cannot handle an array of shape {shape}')
