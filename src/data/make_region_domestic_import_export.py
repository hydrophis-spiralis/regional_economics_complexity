import os
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
raw_dir = os.path.join(project_dir, 'data', 'raw')
interim_dir = os.path.join(project_dir, 'data', 'interim')
external_data_dir = os.path.join(project_dir, 'data', 'external')

country_codes = pd.read_csv(os.path.join(external_data_dir, 'country_table.csv'))
country_map = dict(zip(country_codes.two_digits, country_codes.tri_digits))



def make_export_data_frame(input_filename):
    data = pd.read_excel(input_filename, dtype={'TNVED': 'str'},
                         sheet_name=1)


    data0 = pd.read_excel(input_filename, dtype={'TNVED': 'str'},
                          sheet_name=0)

    data0.TNVED = data0.TNVED.str[:4]
    data.TNVED = data.TNVED.str[:4]

    rudata = data[data.STRANA == 'RU'].copy()

    foreign_export_data = data[data.STRANA != 'RU']
    foreign_export_data = foreign_export_data.groupby(['NAPR', 'PERIOD', 'STRANA', 'TNVED']).STOIM.sum().reset_index()

    foreign_export_data = foreign_export_data[foreign_export_data['STRANA'].isin(country_codes.two_digits)]
    foreign_export_data['STRANA'] = foreign_export_data['STRANA'].map(country_map)

    foreign_export_data['origin'] = 'KLD'
    region_foreign_export_frame = pd.DataFrame({
        'origin': foreign_export_data.origin,
        'dest': foreign_export_data .STRANA,
        'hs07': foreign_export_data .TNVED,
        'export_val': foreign_export_data .STOIM
    })

    region_foreign_export_frame.to_csv(os.path.join(interim_dir, 'region_foreign_export.csv'), index=None)



    merged = pd.merge(rudata, data0, left_on='TNVED', right_on='TNVED')
    ru_data = merged[['NAPR_x', 'PERIOD_x', 'STRANA_x', 'TNVED', 'STOIM_y']]
    ru_data.columns = list(data0.columns)
    ru_data = ru_data.groupby(['NAPR', 'PERIOD', 'STRANA', 'TNVED']).STOIM.sum().reset_index()
    ru_data['origin'] = 'KLD'

    ru_data = ru_data[ru_data.STRANA.isin(country_codes.two_digits)]
    ru_data['STRANA'] = ru_data.STRANA.map(country_map)

    region_domestic_export_frame = pd.DataFrame({
        'origin': ru_data.origin,
        'dest': ru_data.STRANA,
        'hs07': ru_data.TNVED,
        'export_val': ru_data.STOIM
    })

    region_domestic_export_frame.to_csv(os.path.join(interim_dir, 'region_domestic_export.csv'), index=None)
    return region_domestic_export_frame, region_foreign_export_frame


def make_foreign_import_data_frame(input_filename):
    data = pd.read_excel(input_filename, dtype={'TNVED': 'str'}, sheet_name=0)
    data.TNVED = data.TNVED.str[:4]
    data = data.groupby(['NAPR', 'PERIOD', 'STRANA', 'TNVED']).STOIM.sum().reset_index()
    data['dest'] = 'KLD'

    data = data[data.STRANA.isin(country_codes.two_digits)]
    data['STRANA'] = data.STRANA.map(country_map)


    region_foreign_import_frame = pd.DataFrame(
        {
            'origin': data.STRANA,
            'dest': data.dest,
            'hs07': data.TNVED,
            'export_val': data.STOIM
        })

    region_foreign_import_frame.to_csv(os.path.join(interim_dir, 'region_foreign_import.csv'), index=None)
    return region_foreign_import_frame


def make_domestic_import(fn):
    data = pd.read_excel(fn, dtype={'TNVED': 'str'})
    data = data[data.STRANA == 'RU']
    data['dest'] = 'KLD'
    data = data[data.STRANA.isin(country_codes.two_digits)]
    data['STRANA'] = data.STRANA.map(country_map)

    region_domestic_import_frame = pd.DataFrame({
        'origin': data.STRANA,
        'dest': data.dest,
        'hs07': data.TNVED,
        'export_val': data.STOIM
    })
    region_domestic_import_frame.to_csv(os.path.join(interim_dir, 'region_domestic_import.csv'), index=None)
    return region_domestic_import_frame


if __name__ == '__main__':

    region_domestic_export_frame, region_foreign_export_frame = \
        make_export_data_frame(os.path.join(raw_dir, 'Уточненный экспорт  Калининграсдкой области 6-знаков.xlsx'))

    region_foreign_import_frame = make_foreign_import_data_frame(os.path.join(raw_dir, '21.04.ИМПОРТ ЭКСПОРТ 2016.xlsx'))
    region_domestic_import_frame = make_domestic_import(os.path.join(raw_dir, 'Сводный импорт  КО 2016.xlsx'))

