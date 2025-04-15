#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import glob
import os
import time
import pickle
import random
import pickle
from causal_ccm.causal_ccm import ccm
from causal_ccm.pai import pai
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # for showing progress bar in for loops
from torchinfo import summary

# this funciton is to filter data in single FLX csv file
def get_FLX_quality(inputs, threshold, resolution='DD'):
    """quality of FLUXNET2015 site data.

    Args:
        inputs ([type]):
            time series of site data.
        threshold ([type]):
            threshold of length of data. default set 1000 for daily data.
        resolution ([type]):
            time resolution of site data. default set as 'DD'. could be 'HH'.

    Returns:
        quality [type]:
            quality number of site data. 0 for bad site and 1 for good site.
    """
    num_nan = np.sum(np.isnan(np.array(inputs)))
    length = len(inputs)

    # get threshold for specific time resolution
    if resolution == 'DD':
        threshold = threshold
    elif resolution == 'HH':
        threshold = threshold*48
    else:
        raise ValueError('Must daily or half-hour data.')

    if length < threshold:  # control length of inputs.
        quality = 0
    else:
        if num_nan > 0.1*length:  # control length of NaN value.
            quality = 0
        else:
            quality = 1

    return quality



# this funciton is to extract features and labels from single FLX csv file
def get_FLX_inputs(path,
                   label_params,
                   qc_params,
                   resolution,
                   threshold):
    try:
        label = pd.read_csv(path)[label_params]
        qc = pd.read_csv(path)[qc_params]
        duration = pd.read_csv(path)[['TIMESTAMP']]

        # turn -9999 and flag < 1 to NaN.
        label[label == -9999.000] = np.nan
        label[qc < 1] = np.nan

        # Notes: FLX2015 data always have long NaN array at beginning
        #        of soil moisture. Therefore must remove these long NaN
        #        for specific FLX case.
        gap_idx = np.where(~np.isnan(label))[0]  # label

        label = label[gap_idx[0]:gap_idx[-1]]
        duration = duration[gap_idx[0]:gap_idx[-1]]

        # get quality for each FLX site.
        quality = get_FLX_quality(label, threshold=threshold, resolution=resolution)

        if quality == 1:
            # interpolate output if quality is 1.
            label = label.interpolate(method='linear')
        else:
            quality = 0
            label = None
            duration = None
            
    except:
        quality = 0
        duration = None
        label = None

    return label, duration, quality



# this funciton is to split training and test data
def make_train_test_data(feature, label, len_input = 14, len_output = 7, time_lag = 0, test_ratio = 0.2):
  n = len(feature)
  cut_off = int(n * (1 - test_ratio))
  len_output = len_output + time_lag

  x_max = feature.iloc[:cut_off - len_output].max()
  x_min = feature.iloc[:cut_off - len_output].min()

  y_max = label.iloc[len_input:cut_off].max()
  y_min = label.iloc[len_input:cut_off].min()

  feature = (feature - x_min)/(x_max - x_min)
  label = (label - y_min)/(y_max-y_min)

  input = []
  output = []

  for i in range(n - len_input - len_output + 1):
    input.append(feature.iloc[i:i+len_input].to_numpy())
    output.append(label.iloc[i+len_input:i+len_input+len_output].to_numpy())

  input = torch.Tensor(input)
  output = torch.Tensor(output)


  train_x = input[:cut_off-len_output]
  test_x = input[cut_off-len_output:]

  train_y = output[:cut_off-len_output]
  test_y = output[cut_off-len_output:]

  return train_x, train_y[:,-(len_output-time_lag):,:], test_x, test_y[:,-(len_output-time_lag):,:], y_max, y_min




def read_flux(path, threshold = 1):
  label_params=['SWC_F_MDS_1']
  qc_params = []
  qc_params.append(label_params[0]+'_QC')
  label, duration, quality = get_FLX_inputs(
      path=path,
      label_params=label_params,
      qc_params=qc_params,
      resolution='DD',
      threshold=threshold)
  return label, duration, quality




# this function is to preprocess the nc files
def nc_dataset_process(ds, dir_name):
    
    warnings.filterwarnings('ignore')

    
    ds['time'] = ds['time'].values.astype('datetime64[D]')
    
    warnings.filterwarnings('default')
    
    if 'latitude' in ds:
        ds = ds.rename({'latitude': 'lat', 'longitude':'lon'})

    if 'daily_max_temperature' in dir_name:
        ds = ds.rename({'t2m':'max_t2m'})

    if 'daily_mean_temperature' in dir_name:
        ds = ds.rename({'t2m':'mean_t2m'})

    if ds['lon'].values[0] == 0:
        ds['lon'] = ds['lon'] - 180

    if '__xarray_dataarray_variable__' in ds:
        ds = ds.rename({'__xarray_dataarray_variable__':'ws'})
    
    if not ds['lon'].to_index().is_monotonic_increasing:
        ds = ds.reindex(lon=ds.lon[::-1])
    
    if not ds['lat'].to_index().is_monotonic_increasing:
        ds = ds.reindex(lat=ds.lat[::-1])

    return ds


def get_var_name():
    return ['d2m', 'cape', 'max_t2m','mean_t2m','r','e','z','pr','stl1','swvl1','sp','ws']

def get_dir2var():
    file_pattern = ['2m_dewpoint_temperature_ERA5_0.25deg_dailymean_1979-2017',

    'convective_available_potential_energy_ERA5_0.25deg_dailymean_1979-2017',
    'daily_max_temperature_ERA5_0.25deg_daily_1979-2017',
    'daily_mean_temperature_ERA5_0.25deg_daily_1979-2017',
    'era5_relative_humidity_1khpa_0.25deg_dailymean_1991-2014',
    'evaporation_ERA5_0.25deg_dailymean_1979-2018',
    'geopotential_height_500hPa_ERA5_0.25deg_dailymean_1979-2018',
    'precipitation_ERA5_0.25deg_daily_1955-2020',
    'soil_temperature_layer_1',
    'soil_water_layer_1',
    'surface_pressure_ERA5_0.25deg_dailymean_1979-2018',
    'wind_speed_10m']

    var_name = ['d2m', 'cape', 'max_t2m','mean_t2m','r','e','z','pr','stl1','swvl1','sp','ws']

    return dict(zip(file_pattern, var_name))



# this function is to extract the era5 data of all variables around the monitoring station, time range [start_time - 60 days, end_time]
def get_nc_one_site(site_name, grid):
    # grid: to extract the nearest grib * grib data points of era5 
    dir2var = get_dir2var()

    file_name = 'input/' + site_name + '_grid'+ str(grid) + '.npy'
    if os.path.exists(file_name):
        nc = np.load(file_name)
        return nc

    result_array = []

    # area slice
    [lon, lat] = lon_lat[site_name]
    lats =  [(lat//0.25 - grid//2 + 1) * 0.25,(lat//0.25 + grid//2) * 0.25]
    lons = [(lon//0.25 - grid//2 + 1) * 0.25, (lon//0.25 + grid//2) * 0.25]
    
    # time slice
    start_time = durations[site_name]['TIMESTAMP'].iloc[0] - np.timedelta64(60, 'D')
    end_time = durations[site_name]['TIMESTAMP'].iloc[-1]

    file_pattern = os.listdir('D:/data/era5/')
    for i in file_pattern: 
        path = 'D:/data/era5/' + i + '/*.nc'
        var = dir2var[i]
        ds = xr.open_mfdataset(path,  parallel=True, chunks={"time": 10})
        ds = nc_dataset_process(ds, i)
        # print(ds)
        aoi = ds.sel(lat = slice(*lats), lon = slice(*lons), time = slice(start_time , end_time))
        # print(aoi)
        result_array.append(aoi[var].values)
    
    result_array = np.array(result_array)
    np.save(file_name, result_array)    
    
    return result_array



# this function is to extract the era5 data of all variables around the monitoring stations, for all given stations in the desired period.  
def get_nc_site_set(site_set, seq_len, pre_hor, grid):
    '''
    seq_len: lenth of sequence, i.e., no. of days of input feature
    pre_hor: predictive horizon, i.e., the lag between input feature and output label
    grid: to extract the nearest grib * grib data points of era5 
    example: using data on day 0, day 1 to predict feature on day 3, then seq_len = 2, pre_hor = 1
    '''
    
    era5_start = np.datetime64('1991-01-01')
    res = dict()
    for site_name in site_set:
        
        nc = get_nc_one_site(site_name, grid)

        npy_start_time = durations[site_name]['TIMESTAMP'].iloc[0] - np.timedelta64(60, 'D')  
        input_start_time = durations[site_name]['TIMESTAMP'].iloc[0] - np.timedelta64(seq_len + pre_hor - 1, 'D')
        input_end_time = durations[site_name]['TIMESTAMP'].iloc[-1] - np.timedelta64(pre_hor, 'D')
        
        input_start_time = (input_start_time - npy_start_time).days
        input_end_time = (input_end_time - npy_start_time).days + 1
        
        nc_sliced = nc[ : ,input_start_time:input_end_time, : , : ]
        res[site_name] = nc_sliced
            
    return res
    


# this function is to get the means and stds for normalization of data
def get_statistics():
    dir2var = get_dir2var()
    if os.path.exists('temp/mean_log.pck'):
        with open('temp/mean_log.pck', 'rb') as file:  
            mean_log = pickle.load(file)
        with open('temp/std_log.pck', 'rb') as file:  
            std_log = pickle.load(file)
        return mean_log, std_log
    
    mean_log = dict()
    std_log = dict()

    for i in file_pattern: 
        path = 'D:/data/era5/' + i + '/*.nc'
        var = dir2var[i]
        ds = xr.open_mfdataset(path,  parallel=True, chunks={"time": 10})
        ds = nc_dataset_process(ds, i)

        if var not in mean_log:   
            mean_log[var] = ds.sel(time='1991')[var].mean().values
        if var not in std_log:
            std_log[var] = ds.sel(time='1991')[var].std().values
    
    with open('temp/mean_log.pck', "wb") as f:  
        pickle.dump(mean_log, f) 
    
    with open('temp/std_log.pck', "wb") as f:  
        pickle.dump(std_log, f) 
    
    return mean_log, std_log


#### Screen valid stations


def get_valid_site(threshold):

    
    if os.path.exists('temp/valid_site.pck'):
        with open('temp/valid_site.pck', 'rb') as file:  
            valid_site = pickle.load(file)
        with open('temp/labels.pck', 'rb') as file:  
            labels = pickle.load(file)
        with open('temp/durations.pck', 'rb') as file:  
            durations = pickle.load(file)
        with open('temp/lon_lat.pck', 'rb') as file:  
            lon_lat = pickle.load(file)
        return valid_site, labels, durations, lon_lat
    
    site_info = pd.read_excel('label/FLX_AA-Flx_BIF_DD_20200501.xlsx')
    site_id = site_info['SITE_ID'].unique()
    valid_site = []
    labels = dict()
    durations = dict()
    lon_lat = dict()

    for name in site_id:
        path = glob.glob('label/FLX_' + name + '*.csv')
        label, duration, quality = read_flux(path = path[0], threshold = 1 )
        if quality != 0:
            valid_site.append(name)
            labels[name] = label
            duration['TIMESTAMP'] = pd.to_datetime(duration['TIMESTAMP'], format='%Y%m%d')
            durations[name] = duration
            
            lat = site_info.loc[(site_info['SITE_ID'] == name) & (site_info['VARIABLE'] == 'LOCATION_LAT'),'DATAVALUE'].iloc[0]
            lon = site_info.loc[(site_info['SITE_ID'] == name) & (site_info['VARIABLE'] == 'LOCATION_LONG'),'DATAVALUE'].iloc[0]
            lon_lat[name] = [float(lon), float(lat)]
            
    with open('temp/valid_site.pck', "wb") as f:  
        pickle.dump(valid_site, f) 
    with open('temp/labels.pck', "wb") as f:  
        pickle.dump(labels, f)  
    with open('temp/durations.pck', "wb") as f:  
        pickle.dump(durations, f) 
    with open('temp/lon_lat.pck', "wb") as f:  
        pickle.dump(lon_lat, f) 
        
    return valid_site, labels, durations, lon_lat



# #### Torch dataset & dataloader


class TimeseriesDataset(torch.utils.data.Dataset):
    '''
    this funcion is to generate torch dataset for our case,
    considering the input data is 60 steps ahead of the feature   
    '''
    def __init__(self, X, y, cell_state = None, seq_len=15, pre_hor = 7):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.pre_hor = pre_hor
        self.cell_state = cell_state

    def __len__(self):
        return self.y.__len__()

    def __getitem__(self, index):
        if self.cell_state == None:
            return (self.X[index + 61 - self.seq_len - self.pre_hor: index + 61 - self.pre_hor], self.y[index])
        else:
            return (self.X[index + 61 - self.seq_len - self.pre_hor: index + 61 - self.pre_hor],self.cell_state, self.y[index])




def data_read(site_name):
    '''
    this function is to read and normalize the data for model
    '''

    var_name = get_var_name()
    mean_log, std_log = get_statistics()
    valid_site, labels, durations, lon_lat = get_valid_site(threshold = 1) 

    x = np.load('input/'+ site_name + '_grid10.npy')

    for i, name in enumerate(var_name):
        x[i] = (x[i] - mean_log[name]) / std_log[name]

    x = np.moveaxis(x, 0, 1)
    y = np.array(labels[site_name]['SWC_F_MDS_1'])

    y = y/100
    y = y.astype(np.float32)
    
    return x, y






# #### CCM calculation



def ccm_cal(x, y, L):
    '''
    L is the traning set lenth
    Only 4 central data points involved cmm calculation to ensure efficiency    
    '''
    tau = 1 # time lag
    E = 2 # shadow manifold embedding dimensions

    _x = x[60:L+60]
    _x = _x[:,:,4:6,4:6]
    _x = _x.reshape(L,-1)
    _y = y[:L]

    cor_ = []
    p_v = []
    for i in range(_x.shape[-1]):
        #t000 = time.time()
        ccm1 = ccm(_x[:,i], _y, tau, E, L)
        cor_.append(ccm1.causality()[0])
        p_v.append(ccm1.causality()[1])
        #print(time.time() - t000)
    cell_state = cor_ + p_v
    cell_state = torch.tensor(cell_state).to(device)
    cell_state = cell_state.to(torch.float32)
    
    return cell_state


# In[20]:


def get_ccm():
    '''
    this function is to get the cmm at all stations, by loading existing file or calculating one by one
    '''
    try:
        f = open('temp/cell_states.pck', 'rb')
        cell_states = pickle.load(f)
        f.close()
    except: 
        cell_states = dict()
        
        site_num = len(valid_site)
        
        for i, site_name in enumerate(valid_site):
            print(site_name, i+1, '/', site_num)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
            x, y = data_read(site_name)
            train_len = int(len(y) * 0.7)
            cell_state = ccm_cal(x, y, train_len)
            cell_states[site_name] = cell_state
        
        with open('temp/cell_states.pck', "wb") as f:  
            pickle.dump(cell_states, f)
    return cell_states
