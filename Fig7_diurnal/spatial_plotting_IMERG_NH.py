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

import cartopy.feature as cfeature
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
#    ds = ds.swap_dims({'time': 'local_solar_time'})  # Add this line
    
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
        lat_min=-5
        lat_max=35
        lon_min=-20
        lon_max=50
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
    var_08_diur          = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_diur          = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_diur          = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_imerg_diur   = var_obs_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

###

    var_08_time_index    = var_08_diur.argmax(dim='time')
    peak_time_08 = time[var_08_time_index]

    var_06_time_index    = var_06_diur.argmax(dim='time')
    peak_time_06 = time[var_06_time_index]

    var_05_time_index    = var_05_diur.argmax(dim='time')
    peak_time_05 = time[var_05_time_index]

    var_obs_imerg_time_index    = var_obs_imerg_diur.argmax(dim='time')
    peak_time_obs_imerg = time[var_obs_imerg_time_index]


    # Then convert to local solar time at each grid point
    local_peak_time_08_map = var_08_diur['local_solar_time'].isel(time=peak_time_08)
    local_peak_time_06_map = var_06_diur['local_solar_time'].isel(time=peak_time_06)
    local_peak_time_05_map = var_05_diur['local_solar_time'].isel(time=peak_time_05)
    local_peak_time_obs_imerg_map = var_obs_imerg_diur['local_solar_time'].isel(time=peak_time_obs_imerg)




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
    if i == 1:
             i = 1

    elif i == 2: 
             i = 5

    elif i == 3: 
             i = 9

    ax1 = fig.add_subplot(3,4,i, projection=proj)
    local_peak_time_obs_imerg_map.plot.pcolormesh(ax=ax1, transform=ccrs.PlateCarree(), cmap=custom_cmap,  add_colorbar=False, add_labels=False)
    
    ax2 = fig.add_subplot(3,4,i+1, projection=proj)
    local_peak_time_08_map.plot.pcolormesh(ax=ax2, transform=ccrs.PlateCarree(), cmap=custom_cmap,   add_colorbar=False, add_labels=False)   
 
    ax3 = fig.add_subplot(3,4,i+2, projection=proj)
    local_peak_time_06_map.plot.pcolormesh(ax=ax3, transform=ccrs.PlateCarree(), cmap=custom_cmap,  add_colorbar=False, add_labels=False)    

    ax4 = fig.add_subplot(3,4,i+3, projection=proj) 
    local_peak_time_05_map.plot.pcolormesh(ax=ax4, transform=ccrs.PlateCarree(), cmap=custom_cmap,  add_colorbar=False, add_labels=False)


    all_axes = []
    if i == 9:
           all_axes.extend([ax1, ax2, ax3, ax4])

           cbar = fig.colorbar(all_axes[0].collections[0], ax=all_axes,
                  pad=0.02, fraction=0.08,aspect=40,extend='both',orientation='horizontal')

           cbar = ax1.collections[0].colorbar
           cbar.set_ticks(tick_doys)  # Set ticks at 10-day intervals (DOY)
           #cbar.set_ticklabels(tick_labels)
           cbar.set_label('Hour (LST)')
           


    ax1.coastlines(resolution='10m',color='black',linewidth=2)
    ax1.add_feature(cfeature.BORDERS, linewidth=1.5)
    ax1.set_title('IMERG (Reference)',fontweight='bold', fontsize=10)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.xaxis.set_tick_params(labelsize=2)
    ax1.yaxis.set_tick_params(labelsize=2)
    ax1.set_aspect('auto')
    gls = ax1.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False

    ax2.coastlines(resolution='10m',color='black',linewidth=2)
    ax2.add_feature(cfeature.BORDERS, linewidth=1.5)
    ax2.set_title('ICON (10KM)',fontweight='bold', fontsize=10)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.xaxis.set_tick_params(labelsize=2)
    ax2.yaxis.set_tick_params(labelsize=2)
    ax2.set_aspect('auto')
    gls = ax2.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False


    ax3.coastlines(resolution='10m',color='black',linewidth=2)
#    ax2.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax3.add_feature(cfeature.BORDERS, linewidth=1.5)
    ax3.set_title('ICON (40KM)',fontweight='bold', fontsize=10)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.xaxis.set_tick_params(labelsize=2)
    ax3.yaxis.set_tick_params(labelsize=2)
    ax3.set_aspect('auto')
    gls = ax3.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False



    ax4.coastlines(resolution='10m',color='black',linewidth=2)
#    ax2.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax4.add_feature(cfeature.BORDERS, linewidth=1.5)
    ax4.set_title('ICON (80KM)',fontweight='bold', fontsize=10)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.xaxis.set_tick_params(labelsize=2)
    ax4.yaxis.set_tick_params(labelsize=2)
    ax4.set_aspect('auto')
    gls = ax4.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False

    if i == 1:
            ax1.set_title('(a) IMERG (Reference)',fontweight='bold', fontsize=14)
            ax2.set_title('(b) ICON (10KM)',fontweight='bold', fontsize=14)
            ax3.set_title('(c) ICON (40KM)',fontweight='bold', fontsize=14)
            ax4.set_title('(d) ICON (80KM)',fontweight='bold', fontsize=14)

    if i == 5:
            ax1.set_title('(e) IMERG (Reference)',fontweight='bold', fontsize=14)
            ax2.set_title('(f) ICON (10KM)',fontweight='bold', fontsize=14)
            ax3.set_title('(g) ICON (40KM)',fontweight='bold', fontsize=14)
            ax4.set_title('(h) ICON (80KM)',fontweight='bold', fontsize=14)

    if i == 9:
            ax1.set_title('(i) IMERG (Reference)',fontweight='bold', fontsize=14)
            ax2.set_title('(j) ICON (10KM)',fontweight='bold', fontsize=14)
            ax3.set_title('(k) ICON (40KM)',fontweight='bold', fontsize=14)
            ax4.set_title('(l) ICON (80KM)',fontweight='bold', fontsize=14)



#plt.tight_layout()
plt.savefig('Fig7_diurnal')
fig.savefig("Fig7_Diurnal_Spatial.pdf", bbox_inches='tight', pad_inches=0.1,dpi=50)
