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


##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Expected 1D input for filtering along 'time'")
    nyquist = 0.5 * fs
    low = 1 / highcut / nyquist
    high = 1 / lowcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)  # no axis specified for 1D


def apply_bandpass(x, lowcut=2.1, highcut=30, fs=1):
    return xr.apply_ufunc(
        butter_bandpass_filter,
        x,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        kwargs={"lowcut": lowcut, "highcut": highcut, "fs": fs},
        dask="allowed",  # You said you don’t want to use Dask — can change to None if not using Dask
        output_dtypes=[x.dtype]
    )



# base directory where your analysis data is stobrown GPU
# Directory of 10KM simulation 
## NEEDS TO BE CHANGED HERE: 
data_base_dir_08 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/Transport'
model_data_08    =  'u_all_level_rmp_50.nc'

# directory of 40KM simulation
data_base_dir_06 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/Transport'
model_data_06    =  'u_all_level_rmp_50.nc'


# Directory for 80KM simulation

data_base_dir_05 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/Transport'
model_data_05    =  'u_all_level_rmp_50.nc'


# Observational Data Sets for ERA5

data_base_dir_obs_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/Transport'
model_data_obs_era5    =  'u_era5_all_levels.nc'



# netCDF variable name of variable to plot
var='qv','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_obs_era5    = os.path.join(data_base_dir_obs_era5, model_data_obs_era5)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_era5  = xr.open_dataset(model_data_dir_obs_era5)


# LON & LAT

lon = ds_08['lon']
lat = ds_08['lat']


##

# Color Scales Range
a = [0,.01,.1,.25,.5,1,1.5,2,3,4,6,8,10,15,20,30]
a = [0,.005,.01,.15,.25,0.5,1,2,3,4,5,6,7,8,20,30]
a = [-30,-20,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,20,30]
bounds = np.linspace(0,100,10)
#bounds = np.arange(1,10.1,1)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)


# The plotting call from here 

fig = plt.figure(figsize=(16, 12), constrained_layout=True)

colors = ['r', 'g', 'b', 'k']
labels = ['ICON-10KM', 'ICON-40KM', 'ICON-80KM', 'OBS (MSWEP)']
Titles = ['Asian Summer Monsoon', 'North African Monsoon', 'North American Monsoon']

reg = 'South_Asia','North Africa','North America'

reg = 'North Africa'

 lat_min=-10
 lat_max=30
 lon_min=-10
 lon_max=10
 plot_name='North Africa'
 i=1


for x in reg:

    if x == 'South_Asia':
    # Select a Region (Monsoon Core Region)
    #South Asia
        lat_min=10
        lat_max=25
        lon_min=72
        lon_max=85
        plot_name='South_Asia'
        i=1

    elif x == 'North Africa':
        lat_min=-10
        lat_max=30
        lon_min=-10
        lon_max=10
        plot_name='North Africa'
        i=1

    elif x == 'North America':
        lat_min=-20
        lat_max=10
        lon_min=-75
        lon_max=-40
        plot_name='North America'
        i=3
   
    elif x == 'North_Africa_Variance':
        lat_min=-10
        lat_max=30
        lon_min=-10
        lon_max=10
        plot_name='North Africa'
        i=2



 
    plot_name = 'NH_' + 'U'
    print(plot_name)

# Read the Variable from Here.  
    var_08_u = ds_08['u']
    var_06_u = ds_06['u']
    var_05_u = ds_05['u'] 
    var_obs_era5_u = ds_obs_era5['u']

# Select Lat & Lon focussing on the tropics 
    var_08_u = var_08_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_u = var_06_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_u = var_05_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_era5_u = var_obs_era5_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select the Seasons to Caluculate the Monsoon Onset & Do Pentad Centered over a day!      
    var_08_u_summer = var_08_u.sel(time=var_08_u.time.dt.month.isin([6,7,8,9]))
    var_08_u_summer_clim = var_08_u_summer.mean(dim=["lon", "time"])
    var_08_u_summer_clim = var_08_u_summer_clim.sortby('plev', ascending=False)


    var_06_u_summer = var_06_u.sel(time=var_06_u.time.dt.month.isin([6,7,8,9]))
    var_06_u_summer_clim = var_06_u_summer.mean(dim=["lon", "time"])
    var_06_u_summer_clim = var_06_u_summer_clim.sortby('plev', ascending=False)



    var_05_u_summer = var_05_u.sel(time=var_05_u.time.dt.month.isin([6,7,8,9]))
    var_05_u_summer_clim = var_05_u_summer.mean(dim=["lon", "time"]) 
    var_05_u_summer_clim = var_05_u_summer_clim.sortby('plev', ascending=False)


    var_obs_era5_u_summer = var_obs_era5_u.sel(time=var_obs_era5_u.time.dt.month.isin([6,7,8,9]))
    var_obs_era5_u_summer_clim = var_obs_era5_u_summer.mean(dim=["lon", "time"])
    var_obs_era5_u_summer_clim = var_obs_era5_u_summer_clim.sortby('plev', ascending=False)

# 
#    cmap = mpl.colormaps['Spectral_r']
    cmap = mpl.colormaps['bwr']

#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
    
    if i == 1:
             i = 1
             arrow_scale=1
             bounds = np.linspace(-40,20,20)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    elif i == 2: 
             i = 5
             arrow_scale=1
             bounds = np.arange(-20,20.1,0.5)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    elif i == 3: 
             i = 9
             arrow_scale=1
             bounds = np.linspace(-20,10,20)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)


#    ax2 = fig.add_subplot(2,4,i+1)
     # Plot countours for 10KM, 40KM and 80KM
#    var_08_u_summer_clim.plot.contourf(ax=ax2, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])

#    var_08_u_summer_clim.plot.contour(ax=ax2,levels=bounds,colors="black",linewidths=0.8,add_labels=False)


# 3. Add quiver — optionally slice to avoid overcrowding

#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5) 
    
#    ax3 = fig.add_subplot(2,4,i+2) 
#    var_06_u_summer_clim.plot.contourf(ax=ax3, cmap=cmap, x="lat",y="plev", norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
#    var_06_u_summer_clim.plot.contour(ax=ax3,levels=bounds,colors="black",linewidths=0.8,add_labels=False)



#    ax4 = fig.add_subplot(2,4,i+3)
#    var_05_u_summer_clim.plot.contourf(ax=ax4, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
#    var_05_u_summer_clim.plot.contour(ax=ax4,levels=bounds,colors="black",linewidths=0.8,add_labels=False)
#
#    ax1 = fig.add_subplot(2,4,i)
#    var_obs_era5_u_summer_clim.plot.contourf(ax=ax1, cmap=cmap, x="lat",y="plev",norm=norm, cbar_kwargs={'pad': 0.02},levels=[-8, -7, -6])
#    var_obs_era5_u_summer_clim.plot.contour(ax=ax1,levels=bounds,colors="black",linewidths=0.8,add_labels=False)
#


    ax2 = fig.add_subplot(2,4,i+1)
# Plot contours for 10KM, 40KM and 80KM
    cf2 = var_08_u_summer_clim.plot.contourf(ax=ax2, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
    cs2 = var_08_u_summer_clim.plot.contour(ax=ax2,levels=bounds,colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
    ax2.clabel(cs2, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
    ax2.axvline(x=-5, color='red', linewidth=1.5, linestyle='--')
    ax2.axvline(x=9.5, color='red', linewidth=1.5, linestyle='--')
    ax2.axvline(x=10, color='black', linewidth=1.5, linestyle='--')
    ax2.axvline(x=18, color='black', linewidth=1.5, linestyle='--')

    ax3 = fig.add_subplot(2,4,i+2)
    cf3 = var_06_u_summer_clim.plot.contourf(ax=ax3, cmap=cmap, x="lat",y="plev", norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
    cs3 = var_06_u_summer_clim.plot.contour(ax=ax3,levels=bounds,colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
    ax3.clabel(cs3, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
    ax3.axvline(x=-5, color='red', linewidth=1.5, linestyle='--')
    ax3.axvline(x=9.5, color='red', linewidth=1.5, linestyle='--')
    ax3.axvline(x=10, color='black', linewidth=1.5, linestyle='--')
    ax3.axvline(x=18, color='black', linewidth=1.5, linestyle='--')

    ax4 = fig.add_subplot(2,4,i+3)
    cf4 = var_05_u_summer_clim.plot.contourf(ax=ax4, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
    cs4 = var_05_u_summer_clim.plot.contour(ax=ax4,levels=bounds,colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
    ax4.clabel(cs4, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
    ax4.axvline(x=-5, color='red', linewidth=1.5, linestyle='--')
    ax4.axvline(x=9.5, color='red', linewidth=1.5, linestyle='--')
    ax4.axvline(x=10, color='black', linewidth=1.5, linestyle='--')
    ax4.axvline(x=18, color='black', linewidth=1.5, linestyle='--')

    ax1 = fig.add_subplot(2,4,i)
    cf1 = var_obs_era5_u_summer_clim.plot.contourf(ax=ax1, cmap=cmap, x="lat",y="plev",norm=norm, cbar_kwargs={'pad': 0.02},levels=[-8, -7, -6])
    cs1 = var_obs_era5_u_summer_clim.plot.contour(ax=ax1,levels=bounds,colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
    ax1.clabel(cs1, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
    ax1.axvline(x=-5, color='red', linewidth=1.5, linestyle='--')
    ax1.axvline(x=9.5, color='red', linewidth=1.5, linestyle='--')
    ax1.axvline(x=10, color='black', linewidth=1.5, linestyle='--')
    ax1.axvline(x=18, color='black', linewidth=1.5, linestyle='--')

# Add coast line 

# Title and axis labels
    

#    ax4.coastlines(resolution='10m',color='black',linewidth=2)
    ax1.set_title('ERA5',fontweight='bold', fontsize=10)
    ax1.set_xlabel('Latitude')
    ax1.set_ylabel('Pressure Level')
    ax1.xaxis.set_tick_params(labelsize=2)
    ax1.yaxis.set_tick_params(labelsize=2)
    ax1.set_aspect('auto')
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)
    ax1.invert_yaxis()
 
    
    ax2.set_title('ICON (10KM)',fontweight='bold', fontsize=10)
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Pressure Level')
    ax2.xaxis.set_tick_params(labelsize=2)
    ax2.yaxis.set_tick_params(labelsize=2)
    ax2.set_aspect('auto')
    ax2.xaxis.set_tick_params(labelsize=10)
    ax2.yaxis.set_tick_params(labelsize=10)
    ax2.invert_yaxis()

    ax3.set_title('ICON (40KM)',fontweight='bold', fontsize=10)
    ax3.set_xlabel('Latitude')
    ax3.set_ylabel('Pressure Level')
    ax3.xaxis.set_tick_params(labelsize=2)
    ax3.yaxis.set_tick_params(labelsize=2)
    ax3.set_aspect('auto')
    ax3.xaxis.set_tick_params(labelsize=10)
    ax3.yaxis.set_tick_params(labelsize=10)
    ax3.invert_yaxis()

#    ax3.coastlines(resolution='10m',color='black',linewidth=2)
#    ax3.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax4.set_title('ICON (80KM)',fontweight='bold', fontsize=10)
    ax4.set_xlabel('Latitude')
    ax4.set_ylabel('Pressure Level')
    ax4.xaxis.set_tick_params(labelsize=2)
    ax4.yaxis.set_tick_params(labelsize=2)
    ax4.set_aspect('auto')
    ax4.xaxis.set_tick_params(labelsize=10)
    ax4.yaxis.set_tick_params(labelsize=10)
    ax4.invert_yaxis()

#plt.tight_layout()
plt.savefig(plot_name)
fig.savefig("African_Waves.pdf", bbox_inches='tight', pad_inches=0.1)
