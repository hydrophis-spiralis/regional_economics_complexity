import numpy as np
import pandas as pd
from ecomplexity import ecomplexity
from ecomplexity import proximity


def calc_complexity(data:pd.DataFrame):
    trade_cols = {'time': 'year', 'loc': 'origin', 'prod': 'hs07', 'val': 'export_val'}
    data = data[['origin', 'hs07', 'export_val', 'year']]

    cdata = ecomplexity(data, trade_cols)
    prox_df = proximity(data, trade_cols)

    return cdata, prox_df
