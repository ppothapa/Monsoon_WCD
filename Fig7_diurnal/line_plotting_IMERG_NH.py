#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
description         Simple template script to read in some 2D netCDF data, perform
                an average and plot the result as a .jpg figure.
author                  Christoph Heim  && modified by Praveen Kumar Pothapakula
date created    20.05.2022
"""
###############################################################################
# system stuff for e.g. I/O
import os
# numpy for matrices
import numpy as np
# xarray for netcdf data
import xarray as xr
# pandas for smart data analysis
import pandas as pd
# plotting
import matplotlib.pyplot as plt
# for geographic plots
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# handling of dates and time
from datetime import datetime, timedelta
# additional stuff for I/O
from pathlib import Path
import cartopy as cart

import matplotlib.colors as colors

import scipy.ndimage

from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm

from matplotlib.colors import ListedColormap





def convert_dataset_time_to_local_solar(ds):
    """
    Convert the time coordinate of the dataset from UTC to local solar time
    """
    # Calculate local solar hour
    utc_hour_decimal = ds['time'].dt.hour + ds['time'].dt.minute/60 + ds['time'].dt.second/3600
    lon_hours = ds['lon'] / 15.0
    local_solar_hour = (utc_hour_decimal + lon_hours) % 24
    
    # Create new time coordinate with local solar time
    # Keep the same date but change the hour to local solar hour
    local_time = ds['time'].copy()
    
    # Update the dataset's time coordinate
    ds = ds.assign_coords(local_solar_time=local_solar_hour)
    
    return ds




##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

# base directory where your analysis data is stobrown GPU
# Directory of 10KM simulation 

data_base_dir_08 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/tot_prec'

data_base_dir_06 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/tot_prec'

data_base_dir_05 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/tot_prec'

data_base_dir_obs ='/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/IMERG/sub_daily/Final_Data/merged'


model_data_08    =  'tot_prec_original_JJAS_diurnal.nc'

model_data_06    =  'tot_prec_original_JJAS_diurnal.nc'

model_data_05    =  'tot_prec_original_JJAS_diurnal.nc'

model_data_obs_imerg    =  'IMERG_2007_2016_hourly_JJAS_diurnal.nc'



# netCDF variable name of variable to plot
var='tot_prec','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_obs_imerg    = os.path.join(data_base_dir_obs, model_data_obs_imerg)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_imerg  = xr.open_dataset(model_data_dir_obs_imerg)
###

ds_08 = convert_dataset_time_to_local_solar(ds_08)
ds_06 = convert_dataset_time_to_local_solar(ds_06)
ds_05 = convert_dataset_time_to_local_solar(ds_05)
ds_obs_imerg = convert_dataset_time_to_local_solar(ds_obs_imerg)


##
bounds = np.arange(0,16.1,1)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

# The plotting call from here 

fig = plt.figure(figsize=(12, 10), constrained_layout=True)

reg = 'South_Asia','North Africa','North America'

for x in reg:

    if x == 'South_Asia':
    # Select a Region
    #South Asia
        lat_min=0
        lat_max=40
        lon_min=65
        lon_max=95
        plot_name='South_Asia'
        i=1

    elif x == 'North Africa':
        lat_min=-10
        lat_max=30
        lon_min=-40
        lon_max=60
        plot_name='North Africa'
        i=2

    elif x == 'North America':
        lat_min=0
        lat_max=60
        lon_min=-130
        lon_max=-40
        plot_name='North America'
        i=3
    
    # Plot Name and Name of the Variable for Plotting.
    plot_var_key = 'tot_prec'
    print(plot_var_key)
    plot_name = 'Summer' + 'Precipitation'
    print(plot_name)

# Read the Variable from Here.  
    var_08 = ds_08["tot_prec"]
    var_06 = ds_06["tot_prec"]
    var_05 = ds_05["tot_prec"] 
    var_obs_imerg = ds_obs_imerg['precipitation']

#   
    time  = (ds_08['time'].dt.hour + 0 ) % 24 # Replace 'time' with the actual time dimension name

# Select Lat & Lon focussing on the tropics 
    line_08 = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max)).mean(dim=['lat','lon'])
    line_06 = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max)).mean(dim=['lat','lon'])
    line_05 = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max)).mean(dim=['lat','lon'])
    line_imerg = var_obs_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max)).mean(dim=['lat','lon'])


######

    cmap = mpl.colormaps['Spectral_r']
    tick_doys = np.arange(0, 23, 1)  # 10-day intervals within the March-July range
    norm = BoundaryNorm(tick_doys, cmap.N)

    colors = [
    '#081d58', '#10316b', '#18457d', '#225ea8',  # Deep Blue to Blue
    '#2879b9', '#3498db', '#41b6c4', '#5ecfa6',  # Blue to Teal/Green
    '#7fcdbb', '#a6d96a', '#c7e9b4', '#d9ef8b',  # Greens
    '#fee08b', '#fdd835', '#fed976', '#ffb74d',  # Yellows to Light Orange
    '#fc8d59', '#f46d43', '#e34a33', '#d73027',  # Orange to Red
 #   '#bd0026', '#a80023', '#800026', '#4d0015'   # Red to Deep Red
    '#225ea8', '#18457d', '#10316b', '#081d58'   #  Blue to Deep Blue

             ]

    colors = [
    '#4b0082', '#6a0dad', '#8a2be2', '#9370db',  # Distinct purples for night (00-04)
    '#000080', '#0000cd', '#1e90ff', '#87cefa',  # Blues for early morning (04-08)
    '#e0ffff', '#f0f8ff', '#f5f5f5', '#fffacd',  # Very light colors for late morning (08-12)
    '#fff44f', '#ffd700', '#ffa500', '#ff8c00',  # Yellows/oranges for afternoon (12-16)
    '#ff6347', '#ff4500', '#dc143c', '#b22222',  # Reds for evening (16-20)
    '#8b0000', '#800080', '#4b0082'              # Dark reds/purples for night (20-24)
     ]




    # Create a ListedColormap
    custom_cmap = ListedColormap(colors)


#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
   
    all_axes = [] 
#    if i == 1:
#             i = 1

#    elif i == 2: 
#             i = 5
#
#    elif i == 3: 
#             i = 9

    ax = fig.add_subplot(3,1,i, projection=proj)
    ax.plot(time,line_08, color="red", label="ICON (10KM)",  linewidth=2.5)
    ax.plot(time,line_06, color="green", label="ICON (40KM)",linewidth=2.5)
    ax.plot(time,line_05, color="blue", label="ICON (80KM)",linewidth=2.5)
    ax.plot(time,line_imerg, color="black", label="IMERG",linewidth=2.5)    


#plt.tight_layout()
#plt.savefig(plot_name)
fig.savefig("Fig7_Dir_line.pdf", bbox_inches='tight', pad_inches=0.1,dpi=100)
