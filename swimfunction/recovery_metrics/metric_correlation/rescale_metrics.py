''' Rescales metric values in the dataframe
using StandardScaler
'''
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import numpy
import pandas

def rescale_df(df: pandas.DataFrame) -> Tuple[pandas.DataFrame, StandardScaler]:
    ''' Applies StandardScaler to all columns with number dtype, except 'fish', 'group', and 'assay'.
    '''
    scaled_df = df.copy()
    scaler = StandardScaler()
    columns = [
        c for c in scaled_df.columns \
            if numpy.issubdtype(scaled_df[c].dtype, numpy.number) \
                and (c not in ['fish', 'group', 'assay'])]
    scaled_df[columns] = scaler.fit_transform(scaled_df[columns])
    return scaled_df, scaler
