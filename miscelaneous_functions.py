import os
import pandas as pd
import openpyxl as xl
from import_excel import import_excel
from ren_ninja_api import fetch_and_average_data_ren_ninja, get_regular_coordinates
import glob
import geopandas as gpd

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")

def stack_time_series(excel_file):
    wb = xl.load_workbook(excel_file)
    df=pd.DataFrame()
    for sheet in wb.worksheets:
        df_to_add=pd.read_excel(excel_file,shett_name=sheet.title)
        df = pd.concat([df,df_to_add], axis=1)
    return df

def fill_missing_values(start_date, end_date, data, method = 'linear'):
    date_range = pd.date_range(start=pd.to_datetime("start_date"), end=pd.to_datetime("end_date"), freq="h")
    missing_dates = date_range[~date_range.isin(data.index)]
    print('There are ' + str(len(missing_dates))+' missing values in the time series.')
    data_reindexed = data.reindex(date_range)
    data_interpolated = data_reindexed.interpolate(method="linear")
    return data_interpolated

def get_pv_data(dpd, dpyn, ndpd, country_name, year, state_name = None):
    points_in_world = gpd.read_file('./optim_mix/grid_with_centroids_states.geojson')
    folder = path_input_data + f'/{country_name}'

    if state_name : 
        partie_name_file = f'grid_locations_averaged_pv_{country_name}_{state_name}_{year}.xlsx'
    else: 
        partie_name_file = f'grid_locations_averaged_pv_{country_name}_{year}.xlsx' 

    chemin_pattern = os.path.join(folder, f'*{partie_name_file}*')
    fichiers_trouves = glob.glob(chemin_pattern)

    if len(fichiers_trouves)==0:
        print('collecting data')
        fetch_and_average_data_ren_ninja(country_name, 1, ['pv'], points_in_world, state = state_name, year=year, save = True, coordinates = mode)

    fichiers_trouves = glob.glob(chemin_pattern)

    file_name = fichiers_trouves[0].split('//',2)[-1]

    if purpose == 'optim':
        norm = None
    elif purpose == 'wavelet':
        norm = 'mean'
    
    PV_ts = import_excel(path_input_data,file_name, 
                                    dpd ,ndpd, dpy, 
                                    interp=True, norm = norm) # interpolate data from dpd to ndpd numper of points per day

    mean_pv = pd.read_excel(path_input_data+file_name).mean().iloc[0]
    print(f'Série temporelle normalisée pour {purpose}: {norm}')

    return PV_ts, mean_pv

def get_wind_data(path_input_data, dpd, dpyn, ndpd, country_name, year, state_name = None, purpose = 'optim'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    points_in_world_path = os.path.join(base_dir, './optim_mix/grid_with_centroids_states.geojson')
    points_in_world = gpd.read_file(points_in_world_path)
    folder = path_input_data + f'/{country_name}'

    if state_name : 
        partie_name_file = f'grid_locations_averaged_wind_{country_name}_{state_name}_{year}.xlsx'
    else: 
        partie_name_file = f'grid_locations_averaged_wind_{country_name}_{year}.xlsx' 

    chemin_pattern = os.path.join(folder, f'*{partie_name_file}*')
    fichiers_trouves = glob.glob(chemin_pattern)

    if len(fichiers_trouves)==0:
        print('collecting data')
        fetch_and_average_data_ren_ninja(country_name, 1, ['wind'], points_in_world, state = state_name, year=year, save = True, coordinates = mode)

    fichiers_trouves = glob.glob(chemin_pattern)

    file_name = fichiers_trouves[0].split('//',2)[-1]

    if purpose == 'optim':
        norm = None
    elif purpose == 'wavelet':
        norm = 'mean'
    
    Wind_ts = import_excel(path_input_data,file_name, 
                                    dpd ,ndpd, dpy, 
                                    interp=True, norm = norm) # interpolate data from dpd to ndpd numper of points per day

    mean_wind = pd.read_excel(path_input_data+file_name).mean().iloc[0]
    print(f'Série temporelle normalisée pour {purpose}: {norm}')

    return Wind_ts, mean_wind