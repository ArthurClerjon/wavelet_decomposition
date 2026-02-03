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

def get_chu_data(country_name, country_code, enr, path_input_data, dpd, dpy, ndpd, region_chu = None, state_name = None, norm= 'mean'):
    """Fetch time series.
    Returns the time series (array) normalized by the mean and the mean (int)"""
    try :
        if enr=='wind':
            suffix_ts = 'wind_onshore_10y'
            suffix_data = 'full_chu_wind_onshore_capacity_weighted'
            # suffix_data = 'chu_wind_onshore_averaged'
            suffix_chu_data = 'chu_wind_onshore_aggregated'
            folder = 'Wind_Onshore/'
        elif enr =='pv':
            suffix_ts = 'pv_fixed_10y'
            suffix_data = 'full_chu_pv_fixed_capacity_weighted'
            # suffix_data = 'chu_pv_fixed_averaged'
            suffix_chu_data = 'pv_fixed_aggregated'
            folder = 'PV_Fixed/'
    except Exception as e:
        print(e)

    if state_name: 
        file_name = f'{country_name}/{country_code}_{region_chu}_{suffix_ts}.xlsx'

        if not os.path.exists(path_input_data+folder+file_name):  
            file_name = f'{country_name}/{country_code}_{region_chu}_{suffix_chu_data}.xlsx'
            df = pd.read_excel(path_input_data+folder+file_name)
            
            time_series = pd.concat([df[col] for col in df.columns], ignore_index=True)
            result_df = pd.DataFrame({'Time Series': time_series})

            # Sauvegarder le résultat dans un nouveau fichier Excel
            result_df.to_excel(path_input_data+folder+f'{country_name}/{country_code}_{region_chu}_{suffix_ts}.xlsx', index=False)
            # print(result_df)
            file_name = f'{country_name}/{country_code}_{region_chu}_{suffix_ts}.xlsx'
    else:
        file_name = f'Countries/{country_code}_full_{suffix_ts}.xlsx'
        # file_name = f'Countries/{country_code}_{suffix_ts}.xlsx'
        if not os.path.exists(path_input_data+folder+file_name):
            file_name = f'Countries/{country_code}_{suffix_data}.xlsx'
            df = pd.read_excel(path_input_data+folder+file_name)
            # print(df)

            time_series = pd.concat([df[col] for col in df.columns], ignore_index=True)
            result_df = pd.DataFrame({'Time Series': time_series})

            # Sauvegarder le résultat dans un nouveau fichier Excel
            result_df.to_excel(path_input_data+folder+f'Countries/{country_code}_full_{suffix_ts}.xlsx', index=False)
            file_name = f'Countries/{country_code}_full_{suffix_ts}.xlsx'
    print(file_name)
    Wind_ts = import_excel(path_input_data+folder,file_name, 
                                    dpd ,ndpd, dpy, 
                                    interp=True, norm = norm) # interpolate data from dpd to ndpd numper of points per day

    mean_wind = pd.read_excel(path_input_data+folder+file_name).mean().iloc[0]

    return Wind_ts, mean_wind