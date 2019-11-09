import os
from pathlib import Path

import pandas as pd
from src.models.ecomplexity_model import calc_complexity

project_dir = Path(__file__).resolve().parents[1]
raw_dir = os.path.join(project_dir, 'data', 'raw')
interim_dir = os.path.join(project_dir, 'data', 'interim')
external_data_dir = os.path.join(project_dir, 'data', 'external')
processed_data_dir = os.path.join(project_dir, 'data', 'processed')


if __name__ == '__main__':
    world_region_data = pd.read_csv(os.path.join(processed_data_dir, '2017_world_data.csv'))

    cdata, prox = calc_complexity(world_region_data)

    cdata.to_csv(os.path.join(processed_data_dir, 'cdata.csv'), index=None)
    prox.to_csv(os.path.join(processed_data_dir, 'prox.csv'), index=None)

