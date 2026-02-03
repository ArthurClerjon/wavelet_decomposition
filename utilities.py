import numpy as np
import pandas as pd
from pulp import *
import pickle
import os
import openpyxl as xl
import plotly.io as pio

pio.renderers.default = 'notebook'

def create_directory(path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(path) if os.path.isfile(path) or '.' in os.path.basename(path) else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

        

def stack_time_series(excel_file):
    """Stack multi-year time series stored in an Excel file with one column per year."""
    wb = xl.load_workbook(excel_file)
    df = pd.DataFrame()
    for sheet in wb.worksheets:
        df_to_add = pd.read_excel(excel_file, sheet_name=sheet.title)
        df = pd.concat([df, df_to_add], axis=1)
    return df

def fill_missing_values(start_date, end_date, data, method='linear'):
    date_range = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="h")
    missing_dates = date_range[~date_range.isin(data.index)]
    print(f'There are {len(missing_dates)} missing values in the time series.')
    data_reindexed = data.reindex(date_range)
    data_interpolated = data_reindexed.interpolate(method=method)
    return data_interpolated

def import_excel(path_input_data, input_file, dpd, ndpd, dpy, interp=True, norm='mean'):
    """
    interp : Interpolate from dpd to ndpd. True or False
    dpd : data per day
    ndpd : new data per day
    Returns a dictionary with stacked time series over the N years of the Excel file. Each year is normalized and interpolated (if true)
    """
    df = pd.read_excel(path_input_data + input_file)
    myarray = df.values
    one_d = myarray.ravel().astype(float)
    mysize = one_d.size
    assert mysize % dpd == 0, 'import_excel : Data does not cover an integer number of days'

    dataperyear = dpd * dpy
    nfullyears = int(mysize / dataperyear)

    for i in range(nfullyears):
        sublist = one_d[i * dataperyear: (i + 1) * dataperyear]
        if norm == 'mean':
            mean = np.mean(sublist)
        elif norm == 'max':
            mean = np.max(sublist)
        else:
            mean = 1
        one_d[i * dataperyear: (i + 1) * dataperyear] = sublist / mean
    one_d = one_d[0:nfullyears * dataperyear]

    if ndpd is not None:
        ndays = nfullyears * dpy
        oldx = np.arange(0, ndays, 1. / dpd)
        newx = np.arange(0, ndays, 1. / ndpd)
        newy = np.interp(newx, oldx, one_d)
    else:
        newy = one_d

    return newy


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

def extract_year_from_ts(time_series, year, ndpd):
    """
    Extracts a single year's worth of data from a time series.

    Parameters:
    -----------
    time_series : array-like
        The input time series data (e.g., hourly, daily, etc.)
    year : int
        The year to extract (0-based index, where year=0 is the first year in the dataset)
    ndpd : int
        Number of data points per day (e.g., 24 for hourly data, 48 for half-hourly data)

    Returns:
    --------
    array-like
        A subset of the time series containing only the requested year's data

    Notes:
    ------
    - Assumes the time series is continuous with no gaps
    - Assumes each year has exactly 365 days (no leap years)
    - Assumes the time series starts at the beginning of year 0
    """

    # Calculate the total number of years in the time series
    # Formula: total_data_points / data_points_per_day / days_per_year
    nyear = len(time_series)/ndpd/365

    # Check if the requested year is within the available data range
    if year < nyear:
        # Calculate the start index for the requested year
        # Formula: year_number * data_points_per_day * days_per_year
        start = year * ndpd * 365

        # Calculate the end index for the requested year
        # Formula: (year_number + 1) * data_points_per_day * days_per_year
        end = (year + 1) * ndpd * 365

        # Extract the subset of the time series for the requested year
        new_time_series = time_series[start:end]

    # Return the extracted year's data
    # Note: If the requested year is out of range, this will return None
    return new_time_series


def format_load_data(country_name, state_name = None):

     # Get the directory of the current file (utilities.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create path names 
    country_codes_path = os.path.join(base_dir, './countries_codes_and_coordinates.csv')
    data_path = os.path.join(base_dir, './input_time_series/All Demand UTC 2015.csv')
    country_path = os.path.join(base_dir,f'./input_time_series/Load/{country_name}/')
    if not os.path.exists(country_path):
        os.makedirs(country_path)
    if state_name:
        file_path = os.path.join(base_dir,f'./input_time_series/Load/{country_name}/{country_name}_{state_name}_demand_Plexos_2015.xlsx')
    else: 
        file_path = os.path.join(base_dir,f'./input_time_series/Load/{country_name}/{country_name}_demand_Plexos_2015.xlsx')

    if os.path.exists(file_path):
        pass
    else: 
        country_codes = pd.read_csv(country_codes_path , sep = ',', index_col = 0)
        data = pd.read_csv(data_path, index_col =0)
        iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
        print(data)
        if state_name:
            print(data.columns.str.endswith(state_name))
            end_column_name = iso_code + '-'+state_name
            column_name = data.columns[data.columns.str.endswith(end_column_name)].item()
            print(column_name)
            data[column_name].to_excel(file_path, index =False)
        else:
            column_name = data.columns[data.columns.str.endswith(iso_code)].item()
            data[column_name].to_excel(file_path, index =False)
        
    return file_path.split('/',2)[-1]

def optimize_multi_year_enr_chu(country_name, Load_ts, PV_ts, Wind_ts, nb_year, mean_load, state_name = None,save_results = False, file_name = None, stock = 10):
    prob = LpProblem(f"myProblem", LpMinimize)

    start = 0
    end = start+nb_year*8760
    Load_ts = Load_ts[start:end]
    PV_ts = PV_ts[start:end]
    Wind_ts = Wind_ts[start:end]

    signal_length = len(Load_ts)

    dt = 1 # hour
    e_factor = stock # max hours of consumption to be stored
    E_max = e_factor*Load_ts.mean()# max capacity of storage (MW)
    stock_efficiency = 0.8 # %
    P_max = 100000 # MW used for the binary variable, voluntraily very high

    # Decision Variables
    x_pv = LpVariable("PV_coefficient", lowBound=0)
    x_wind = LpVariable("Wind_coefficient", lowBound=0)
    ts_dispatchable = LpVariable.dicts('Dispatchable_production', range(signal_length), lowBound=0)
    p_ch = LpVariable.dicts('Pch', range(signal_length), lowBound=0, upBound = P_max)
    p_dech = LpVariable.dicts('Pdech', range(signal_length), lowBound=0, upBound = P_max)
    SOC_ts = LpVariable.dicts('Stock',range(signal_length), lowBound=0, upBound=E_max )
    p_curt = LpVariable.dicts('Curtailment',range(signal_length), lowBound=0)
    dech_active = LpVariable.dicts('Dech_active', range(signal_length), cat='Binary')

    

    # Constraint 1: nodal law
    for t in range(len(Load_ts)):
        prob += x_pv * PV_ts[t] + ts_dispatchable[t] + x_wind * Wind_ts[t]+p_dech[t] == Load_ts[t] +p_ch[t]+p_curt[t]

    # Constraint 2: storage
    for t in range(1, signal_length):
        prob += SOC_ts[t] == SOC_ts[t-1] + (stock_efficiency*p_ch[t]-p_dech[t])*dt
    
        # Binary variable: can't charge and discharge at the same time
        prob += p_ch[t] <= (1-dech_active[t])*P_max 
        prob += p_dech[t] <= (dech_active[t])*P_max

    #TODO : trouver une meilleure contrainte pour p_stock
    prob+= p_ch[0]==0
    prob+=p_dech[0]==0
    prob += SOC_ts[0] == SOC_ts[signal_length-1] #same state of charge at the start and end

    prob += x_pv*PV_ts.sum()+x_wind*Wind_ts.sum() == Load_ts.sum()

    # Fonction objectif
    prob += lpSum(ts_dispatchable[t] for t in range(signal_length))*dt

    prob.solve(GUROBI())

    # Afficher les résultats
    print("Status:", LpStatus[prob.status])
    print("Coefficient optimal pour PV:", x_pv.varValue)
    print("Coefficient optimal pour Wind:", x_wind.varValue)


    # Get results of optimization 
    if not x_pv.varValue:
        x_pv.varValue = 0
    if not x_wind.varValue:
        x_wind.varValue = 0
    optimized_pv = [x_pv.varValue * PV_ts[t] for t in range(signal_length)]
    optimized_wind = [x_wind.varValue * Wind_ts[t] for t in range(signal_length)]
    optimized_dispatchable = [ts_dispatchable[t].varValue for t in range(signal_length)]
    optimized_stock = [SOC_ts[t].varValue for t in range(signal_length)]
    optimized_p_curt = [p_curt[t].varValue for t in range(signal_length)]
    optimized_charge = [p_ch[t].varValue for t in range(signal_length)]
    optimized_discharge = [p_dech[t].varValue for t in range(signal_length)]


    # Calculate energy totals
    E_wind = np.sum(optimized_wind)*mean_load
    E_pv = np.sum(optimized_pv)*mean_load
    E_dispatch = np.sum(optimized_dispatchable)*mean_load
    E_curt = np.sum(optimized_p_curt)*mean_load
    E_loss  = (np.sum(optimized_charge)-np.sum(optimized_discharge))*mean_load
    E_stock = np.sum(optimized_charge)*mean_load
    E_destock = np.sum(optimized_discharge)*mean_load

    country_codes = pd.read_csv('./countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
    'mean_consumption':mean_load,
    'pv_capacity': x_pv.varValue*mean_load, 
    'wind_capacity': x_wind.varValue*mean_load,
    'dispatchable_capacity':np.max(optimized_dispatchable)*mean_load, 
    'E_wind' : E_wind, 
    'E_pv':E_pv, 
    'E_dispatch':E_dispatch, 
    'E_curt':E_curt, 
    'E_loss':E_loss, 
    'E_stock':E_stock,
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch)*100,
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch)*100,
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)*100
}
    if save_results :
        if file_name : 
            filename=f'results/{country_name}/{file_name}'
        elif state_name:
            filename = f'results/{country_name}/optimization_results_capacity_{state_name}_{2008+no_year}_{stock}hSto.pickle'
        else :
            filename = f'results/{country_name}/optimization_results_capacity_{2008+no_year}_{stock}hSto.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def optimize_multi_year_capacity_enr_chu(country_name, Load_ts, PV_ts, Wind_ts, nb_year, mean_load, state_name = None,save_results = False, file_name = None, stock = 10):

    prob = LpProblem(f"myProblem", LpMinimize)

    start = 0
    end = start+nb_year*8760
    Load_ts = Load_ts[start:end]
    PV_ts = PV_ts[start:end]
    Wind_ts = Wind_ts[start:end]

    signal_length = len(Load_ts)

    dt = 1 # hour
    e_factor = 10 # max hours of consumption to be stored
    E_max = e_factor*Load_ts.mean()# max capacity of storage (MW)
    stock_efficiency = 0.8 # %
    P_max = 100000 # MW used for the binary variable, voluntraily very high

    # Decision Variables
    x_pv = LpVariable("PV_coefficient", lowBound=0)
    x_wind = LpVariable("Wind_coefficient", lowBound=0)
    ts_dispatchable = LpVariable.dicts('Dispatchable_production', range(signal_length), lowBound=0)
    p_ch = LpVariable.dicts('Pch', range(signal_length), lowBound=0, upBound = P_max)
    max_dispatchable = LpVariable('Dispatchable_capaicity', lowBound=0)
    p_dech = LpVariable.dicts('Pdech', range(signal_length), lowBound=0, upBound = P_max)
    SOC_ts = LpVariable.dicts('Stock',range(signal_length), lowBound=0, upBound=E_max )
    p_curt = LpVariable.dicts('Curtailment',range(signal_length), lowBound=0)
    dech_active = LpVariable.dicts('Dech_active', range(signal_length), cat='Binary')


    # Constraint 1: nodal law
    for t in range(len(Load_ts)):
        prob += ts_dispatchable[t]<= max_dispatchable
        prob += x_pv * PV_ts[t] + ts_dispatchable[t] + x_wind * Wind_ts[t]+p_dech[t] == Load_ts[t] +p_ch[t]+p_curt[t]

    # Constraint 2: storage
    for t in range(1, signal_length):
        prob += SOC_ts[t] == SOC_ts[t-1] + (stock_efficiency*p_ch[t]-p_dech[t])*dt
    
        # Binary variable: can't charge and discharge at the same time
        prob += p_ch[t] <= (1-dech_active[t])*P_max 
        prob += p_dech[t] <= (dech_active[t])*P_max

    #TODO : trouver une meilleure contrainte pour p_stock
    prob+= p_ch[0]==0
    prob+=p_dech[0]==0
    prob += SOC_ts[0] == SOC_ts[signal_length-1] #same state of charge at the start and end

    prob += x_pv*PV_ts.sum()+x_wind*Wind_ts.sum() == Load_ts.sum()

    # Fonction objectif
    prob += max_dispatchable #+ 0.000001*lpSum(ts_dispatchable[t] for t in range(signal_length))*dt

    prob.solve(GUROBI())

    # Afficher les résultats
    print("Status:", LpStatus[prob.status])
    print("Coefficient optimal pour PV:", x_pv.varValue)
    print("Coefficient optimal pour Wind:", x_wind.varValue)


    # Get results of optimization 
    if not x_pv.varValue:
        x_pv.varValue = 0
    if not x_wind.varValue:
        x_wind.varValue = 0
    optimized_pv = [x_pv.varValue * PV_ts[t] for t in range(signal_length)]
    optimized_wind = [x_wind.varValue * Wind_ts[t] for t in range(signal_length)]
    optimized_dispatchable = [ts_dispatchable[t].varValue for t in range(signal_length)]
    optimized_stock = [SOC_ts[t].varValue for t in range(signal_length)]
    optimized_p_curt = [p_curt[t].varValue for t in range(signal_length)]
    optimized_charge = [p_ch[t].varValue for t in range(signal_length)]
    optimized_discharge = [p_dech[t].varValue for t in range(signal_length)]


    # Calculate energy totals
    E_wind = np.sum(optimized_wind)*mean_load
    E_pv = np.sum(optimized_pv)*mean_load
    E_dispatch = np.sum(optimized_dispatchable)*mean_load
    E_curt = np.sum(optimized_p_curt)*mean_load
    E_loss  = (np.sum(optimized_charge)-np.sum(optimized_discharge))*mean_load
    E_stock = np.sum(optimized_charge)*mean_load
    E_destock = np.sum(optimized_discharge)*mean_load

    country_codes = pd.read_csv('./countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
    'mean_consumption':mean_load,
    'pv_capacity': x_pv.varValue*mean_load, 
    'wind_capacity': x_wind.varValue*mean_load,
    'dispatchable_capacity':np.max(optimized_dispatchable)*mean_load, 
    'E_wind' : E_wind, 
    'E_pv':E_pv, 
    'E_dispatch':E_dispatch, 
    'E_curt':E_curt, 
    'E_loss':E_loss, 
    'E_stock':E_stock,
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch)*100,
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch)*100,
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)*100
}

    if save_results :
        if file_name : 
            filename=f'results/{country_name}/{file_name}'
        elif state_name:
            filename = f'results/{country_name}/optimization_results_capacity_{state_name}_{nb_year}.pickle'
        else :
            filename = f'results/{country_name}/optimization_results_capacity_{nb_year}.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def optimize_enr_chu(country_name, Load_ts, PV_ts, Wind_ts, no_year, mean_load, state_name = None,save_results = False, file_name = None, stock = 10):
    prob = LpProblem(f"myProblem", LpMinimize)

    start = no_year*8760
    end = start+8760
    Load_ts = Load_ts[start:end]
    PV_ts = PV_ts[start:end]
    Wind_ts = Wind_ts[start:end]

    signal_length = len(Load_ts)

    dt = 1 # hour
    e_factor = stock # max hours of consumption to be stored
    E_max = e_factor*Load_ts.mean()# max capacity of storage (MW)
    stock_efficiency = 0.8 # %
    P_max = 100000 # MW used for the binary variable, voluntraily very high

    # Decision Variables
    x_pv = LpVariable("PV_coefficient", lowBound=0)
    x_wind = LpVariable("Wind_coefficient", lowBound=0)
    ts_dispatchable = LpVariable.dicts('Dispatchable_production', range(signal_length), lowBound=0)
    p_ch = LpVariable.dicts('Pch', range(signal_length), lowBound=0, upBound = P_max)
    p_dech = LpVariable.dicts('Pdech', range(signal_length), lowBound=0, upBound = P_max)
    SOC_ts = LpVariable.dicts('Stock',range(signal_length), lowBound=0, upBound=E_max )
    p_curt = LpVariable.dicts('Curtailment',range(signal_length), lowBound=0)
    dech_active = LpVariable.dicts('Dech_active', range(signal_length), cat='Binary')

    

    # Constraint 1: nodal law
    for t in range(len(Load_ts)):
        prob += x_pv * PV_ts[t] + ts_dispatchable[t] + x_wind * Wind_ts[t]+p_dech[t] == Load_ts[t] +p_ch[t]+p_curt[t]

    # Constraint 2: storage
    for t in range(1, signal_length):
        prob += SOC_ts[t] == SOC_ts[t-1] + (stock_efficiency*p_ch[t]-p_dech[t])*dt
    
        # Binary variable: can't charge and discharge at the same time
        prob += p_ch[t] <= (1-dech_active[t])*P_max 
        prob += p_dech[t] <= (dech_active[t])*P_max

    #TODO : trouver une meilleure contrainte pour p_stock
    prob+= p_ch[0]==0
    prob+=p_dech[0]==0
    prob += SOC_ts[0] == SOC_ts[signal_length-1] #same state of charge at the start and end

    prob += x_pv*PV_ts.sum()+x_wind*Wind_ts.sum() == Load_ts.sum()

    # Fonction objectif
    prob += lpSum(ts_dispatchable[t] for t in range(signal_length))*dt

    prob.solve(GUROBI())

    # Afficher les résultats
    print("Status:", LpStatus[prob.status])
    print("Coefficient optimal pour PV:", x_pv.varValue)
    print("Coefficient optimal pour Wind:", x_wind.varValue)


    # Get results of optimization 
    if not x_pv.varValue:
        x_pv.varValue = 0
    if not x_wind.varValue:
        x_wind.varValue = 0
    optimized_pv = [x_pv.varValue * PV_ts[t] for t in range(signal_length)]
    optimized_wind = [x_wind.varValue * Wind_ts[t] for t in range(signal_length)]
    optimized_dispatchable = [ts_dispatchable[t].varValue for t in range(signal_length)]
    optimized_stock = [SOC_ts[t].varValue for t in range(signal_length)]
    optimized_p_curt = [p_curt[t].varValue for t in range(signal_length)]
    optimized_charge = [p_ch[t].varValue for t in range(signal_length)]
    optimized_discharge = [p_dech[t].varValue for t in range(signal_length)]


    # Calculate energy totals
    E_wind = np.sum(optimized_wind)*mean_load
    E_pv = np.sum(optimized_pv)*mean_load
    E_dispatch = np.sum(optimized_dispatchable)*mean_load
    E_curt = np.sum(optimized_p_curt)*mean_load
    E_loss  = (np.sum(optimized_charge)-np.sum(optimized_discharge))*mean_load
    E_stock = np.sum(optimized_charge)*mean_load
    E_destock = np.sum(optimized_discharge)*mean_load

    country_codes = pd.read_csv('./countries_codes_and_coordinates.csv' , sep = ',', index_col = 0)
    iso_code = country_codes.loc[country_name,'Alpha-3 code' ].split(' ')[1]
    
    results = {
    'optimized_pv': optimized_pv,
    'optimized_wind': optimized_wind,
    'optimized_dispatchable': optimized_dispatchable,
    'optimized_stock':optimized_stock, 
    'optimized_charge': optimized_charge, 
    'optimized_discharge':optimized_discharge, 
    'optimized_p_curt' : optimized_p_curt,
    'consumption':np.array(Load_ts),
    'cf_pv': np.mean(PV_ts),
    'cf_wind': np.mean(Wind_ts),
    'mean_consumption':mean_load,
    'pv_capacity': x_pv.varValue*mean_load, 
    'wind_capacity': x_wind.varValue*mean_load,
    'dispatchable_capacity':np.max(optimized_dispatchable)*mean_load, 
    'E_wind' : E_wind, 
    'E_pv':E_pv, 
    'E_dispatch':E_dispatch, 
    'E_curt':E_curt, 
    'E_loss':E_loss, 
    'E_stock':E_stock,
    'E_destock':E_destock, 
    'iso_alpha':iso_code, 
    'share_pv': E_pv/(E_pv+E_wind+E_dispatch)*100,
    'share_wind': E_wind/(E_pv+E_wind+E_dispatch)*100,
    'share_dispatchable': E_dispatch/(E_pv+E_wind+E_dispatch)*100
}
    if save_results :
        if file_name : 
            filename=f'results/{country_name}/{file_name}'
        elif state_name:
            filename = f'results/{country_name}/optimization_results_{state_name}_{2008+no_year}_{stock}hSto.pickle'
        else :
            filename = f'results/{country_name}/optimization_results_{2008+no_year}_{stock}hSto.pickle'
        if not os.path.exists(f'results/{country_name}'):
            os.makedirs(f'results/{country_name}')
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        print(f"Optimization results saved to "+filename)

    return results

def build_predictors_dataset(multi_index, optimization_file, wavelets_file, predictors_file_name, optimization_file_path= './results/', wavelets_file_path = './results_decomposition_coefficient_CHU/wavelets_reconstruction/'):
    df_predictors_states = pd.DataFrame(index = multi_index)

    optimization_results = 'optimization_results'
    optimization_suffix = optimization_file.split('.')[0].split('_')[-1]
    for country, state_name in df_predictors_states.index:
        # print(type(state_name))
        try :
            if isinstance(state_name, float): 
                result_optim = pd.read_pickle(f'{optimization_file_path}{country}/{optimization_file}')
                data_wavelets = pd.read_pickle( f'{wavelets_file_path}wavelets_{country}_{wavelets_file}.pickle')
            else:
                result_optim = pd.read_pickle(f'{optimization_file_path}{country}/{optimization_results}_{state_name}_{optimization_suffix}.pickle')
                data_wavelets = pd.read_pickle( f'{wavelets_file_path}wavelets_{country}_{state_name}_{wavelets_file}.pickle')
            df_predictors_states.loc[(country, state_name),'share_wind'] = result_optim['share_wind']
            df_predictors_states.loc[(country, state_name),'share_pv'] = result_optim['share_pv']
            df_predictors_states.loc[(country, state_name),'share_dispatch'] = result_optim['share_dispatchable']
            df_predictors_states.loc[(country, state_name), 'cf_pv']=result_optim['cf_pv']
            # df_predictors_states.loc[(country, state_name), 'cf_wind']=result_optim['cf_wind']
            # if not isinstance(result_optim['cf_pv'], float):
            #     df_predictors_states.loc[(country, state_name), 'cf_pv']=result_optim['cf_pv'].iloc[0]
            # else: 
            #     df_predictors_states.loc[(country, state_name), 'cf_pv']=result_optim['cf_pv']
            # if not isinstance(result_optim['cf_wind'], float):
            #     df_predictors_states.loc[(country, state_name), 'cf_wind']=result_optim['cf_wind'].iloc[0]
            # else: 
            #     df_predictors_states.loc[(country, state_name), 'cf_wind']=result_optim['cf_wind']

            # Les résultats des décompositions pour chaque signal de conso et de production ont été calculés et sont stcokés dans
            # un fichier .pickle par pays.

            
            # LOAD
            df_predictors_states.loc[(country, state_name), 'beta_year_load'] = np.max(data_wavelets['load_year'])
            df_predictors_states.loc[(country, state_name), 'max_beta_week_load'] = np.max(data_wavelets['load_week'])
            df_predictors_states.loc[(country, state_name), 'max_beta_day_load'] = np.max(data_wavelets['load_day'])
        
            df_predictors_states.loc[(country, state_name), 'mean_beta_week_load'] = np.mean(data_wavelets['load_week'][data_wavelets['load_week']>0])
            df_predictors_states.loc[(country, state_name), 'mean_beta_day_load'] = np.mean(data_wavelets['load_day'][data_wavelets['load_day']>0])

            # PV
            df_predictors_states.loc[(country, state_name), 'beta_year_pv'] = np.max(data_wavelets['pv_year'])
            df_predictors_states.loc[(country, state_name), 'max_beta_week_pv'] = np.max(data_wavelets['pv_week'])
            df_predictors_states.loc[(country, state_name), 'max_beta_day_pv'] = np.max(data_wavelets['pv_day'])
            df_predictors_states.loc[(country, state_name), 'mean_beta_week_pv'] = np.mean(data_wavelets['pv_week'][data_wavelets['pv_week']>0])
            df_predictors_states.loc[(country, state_name), 'mean_beta_day_pv'] = np.mean(data_wavelets['pv_day'][data_wavelets['pv_day']>0])

            # WIND
            df_predictors_states.loc[(country, state_name), 'beta_year_wind'] = np.max(data_wavelets['wind_year'])
            df_predictors_states.loc[(country, state_name), 'max_beta_week_wind'] = np.max(data_wavelets['wind_week'])
            df_predictors_states.loc[(country, state_name), 'max_beta_day_wind'] = np.max(data_wavelets['wind_day'])
            df_predictors_states.loc[(country, state_name), 'mean_beta_week_wind'] = np.mean(data_wavelets['wind_week'][data_wavelets['wind_week']>0])
            df_predictors_states.loc[(country, state_name), 'mean_beta_day_wind'] = np.mean(data_wavelets['wind_day'][data_wavelets['wind_day']>0])
            # SCALAR PRODUCTS
            df_predictors_states.loc[(country, state_name), 'scalar_year_pv_load'] = data_wavelets['scalar_year_pv_load']
            df_predictors_states.loc[(country, state_name), 'scalar_year_wind_load'] = data_wavelets['scalar_year_wind_load']
            df_predictors_states.loc[(country, state_name), 'scalar_day_pv_load'] = data_wavelets['scalar_day_pv_load']
            df_predictors_states.loc[(country, state_name), 'scalar_day_wind_load'] = data_wavelets['scalar_day_wind_load']
            df_predictors_states.loc[(country, state_name), 'scalar_week_pv_load'] = data_wavelets['scalar_week_pv_load']
            df_predictors_states.loc[(country, state_name), 'scalar_week_wind_load'] = data_wavelets['scalar_week_wind_load']
            df_predictors_states.loc[(country, state_name), 'scalar_day_wind_pv'] = data_wavelets['scalar_day_wind_pv']
            df_predictors_states.loc[(country, state_name), 'scalar_year_wind_pv'] = data_wavelets['scalar_year_wind_pv']
            df_predictors_states.loc[(country, state_name), 'scalar_week_wind_pv'] = data_wavelets['scalar_week_wind_pv']

            # WEIGHT TS
            df_predictors_states.loc[(country, state_name), 'weight_pv'] = result_optim['pv_capacity']/result_optim['mean_consumption']
            df_predictors_states.loc[(country, state_name), 'weight_wind'] = result_optim['wind_capacity']/result_optim['mean_consumption']

            # RATIOS
            df_predictors_states.loc[(country, state_name), 'ratio_disp_enr'] = result_optim['E_dispatch']/(result_optim['E_wind']+result_optim['E_pv'])
            df_predictors_states.loc[(country, state_name), 'ratio_pv_enr'] = result_optim['E_pv']/(result_optim['E_wind']+result_optim['E_pv'])
            df_predictors_states.loc[(country, state_name), 'disp_p_max']=result_optim['dispatchable_capacity']/result_optim['consumption'].max()/result_optim['mean_consumption']
        except Exception as e:
            print(f"Error processing {country} and {state_name}: {e}")

        df_predictors_states.to_csv(f"{predictors_file_name}")

    
    return

def compute_scalar_products(country_name,path_to_wavelets, wavelets_file, state_name = None):

    # if state_name : 
    #     if wavelets_file =='':
    #         file_name = f'{path_to_wavelets}/wavelets_{country_name}_{state_name}.pickle'
    #     else: 
    #         file_name = f'{path_to_wavelets}/wavelets_{country_name}_{state_name}_{wavelets_file}.pickle'
    # else :
    #     if wavelets_file == '':
    #         file_name = f'{path_to_wavelets}/wavelets_{country_name}.pickle'
    #     else :
    #         file_name = f'{path_to_wavelets}/wavelets_{country_name}_{wavelets_file}.pickle'
    file_name = os.path.join(path_to_wavelets, wavelets_file)
    print(file_name)
    wavelets = pd.read_pickle(file_name)
    scalar_year_pv = np.dot(wavelets['pv_year'], wavelets['load_year'])
    scalar_year_wind = np.dot(wavelets['wind_year'], wavelets['load_year'])
    scalar_day_pv = np.dot(wavelets['pv_day'], wavelets['load_day'])
    scalar_day_wind = np.dot(wavelets['wind_day'], wavelets['load_day'])
    scalar_week_pv = np.dot(wavelets['pv_week'], wavelets['load_week'])
    scalar_week_wind = np.dot(wavelets['wind_week'], wavelets['load_week'])
    scalar_year_wind_pv = np.dot(wavelets['pv_year'], wavelets['wind_year'])
    scalar_day_wind_pv = np.dot(wavelets['pv_day'], wavelets['wind_day'])
    scalar_week_wind_pv = np.dot(wavelets['pv_week'], wavelets['wind_week'])
    wavelets['scalar_year_pv_load']=scalar_year_pv
    wavelets['scalar_week_pv_load']=scalar_week_pv
    wavelets['scalar_day_pv_load']=scalar_day_pv
    wavelets['scalar_year_wind_load']=scalar_year_wind
    wavelets['scalar_week_wind_load']=scalar_week_wind
    wavelets['scalar_day_wind_load']=scalar_day_wind
    wavelets['scalar_year_wind_pv']=scalar_year_wind_pv
    wavelets['scalar_week_wind_pv']=scalar_week_wind_pv
    wavelets['scalar_day_wind_pv']=scalar_day_wind_pv

    with open(file_name, 'wb') as pickle_file:
        pickle.dump(wavelets, pickle_file)

    return

def dispatchable_constant(load_signal):
    load_zero_centered = load_signal - 1
    e_dispatch = 0
    for e in load_zero_centered:
        if e >0:
            e_dispatch+=e
    return e_dispatch