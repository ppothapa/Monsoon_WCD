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

# base directory where your analysis data is stobrown GPU
# Directory of 10KM simulation 

data_base_dir = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig4_Monsoon_Bias/data_generation'

model_data_08    =  'cli_08_summer.nc'

model_data_06    =  'cli_06_summer.nc'

model_data_05    =  'cli_05_summer.nc'

model_data_obs_imerg    =  'cli_imerg_summer.nc'


# Mask Data Sets
data_base_mask = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig1_domains/original_masks'
mask_08='08_mask_2mm_55_na.nc'
mask_06='06_mask_2mm_55_na.nc'
mask_05='05_mask_2mm_55_na.nc'
mask_era5='era5_mask_2mm_55_na.nc'
mask_imerg='imerg_mask_2mm_55_na.nc'
mask_cpc='cpc_mask_2mm_55.nc'
mask_mswep='mswep_mask_2mm_55_na.nc'



# netCDF variable name of variable to plot
var='tot_prec','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir, model_data_05)
model_data_dir_obs_imerg    = os.path.join(data_base_dir, model_data_obs_imerg)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_imerg  = xr.open_dataset(model_data_dir_obs_imerg)
###

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_imerg = os.path.join(data_base_mask,mask_imerg)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)


##
ds_mask_08 = xr.open_dataset(data_dir_mask_08)
ds_mask_06 = xr.open_dataset(data_dir_mask_06)
ds_mask_05 = xr.open_dataset(data_dir_mask_05)
ds_mask_imerg = xr.open_dataset(data_dir_mask_imerg)
ds_mask_cpc = xr.open_dataset(data_dir_mask_cpc)
ds_mask_era5 = xr.open_dataset(data_dir_mask_era5)
ds_mask_mswep = xr.open_dataset(data_dir_mask_mswep)


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
        lat_max=60
        lon_min=60
        lon_max=150
        plot_name='South_Asia'
        i=1

    elif x == 'North Africa':
        lat_min=0
        lat_max=35
        lon_min=-20
        lon_max=50
        plot_name='North Africa'
        i=2

    elif x == 'North America':
        lat_min=0
        lat_max=32
        lon_min=-120
        lon_max=-75
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

# IMERG DATA 

    # Masks
    mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
    mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
    mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
    mask_era5 = ds_mask_era5["__xarray_dataarray_variable__"]
    mask_imerg = ds_mask_imerg["__xarray_dataarray_variable__"]
    mask_cpc = ds_mask_cpc["__xarray_dataarray_variable__"]
    mask_mswep = ds_mask_mswep["__xarray_dataarray_variable__"]

#    var_obs_imerg = var_obs_imerg.transpose()

## Selecting Lat Lon

    mask_08 = mask_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_06 = mask_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_05 = mask_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_imerg = mask_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_cpc = mask_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_era5 = mask_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_mswep = mask_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select Lat & Lon focussing on the tropics 
    var_08 = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06 = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05 = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_imerg = var_obs_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

#### 
  
    var_08 = var_08 * mask_imerg 
    var_06 = var_06 * mask_imerg
    var_05 = var_05 * mask_imerg
    var_obs_imerg = var_obs_imerg * mask_imerg

######

    cmap = mpl.colormaps['Spectral_r']
    tick_doys = np.arange(0, 16.1, 1)  # 10-day intervals within the March-July range
    norm = BoundaryNorm(tick_doys, cmap.N)

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
    var_obs_imerg.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)

    ax2 = fig.add_subplot(3,4,i+1, projection=proj)
     # Plot countours for 10KM, 40KM and 80KM
    var_08.plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5) 
    
    ax3 = fig.add_subplot(3,4,i+2, projection=proj) 
    var_06.plot.contourf(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)
 
    ax4 = fig.add_subplot(3,4,i+3, projection=proj)
    var_05.plot.contourf(ax=ax4, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,  add_colorbar=False, add_labels=False)
       
    if i == 9: 
           all_axes.extend([ax1, ax2, ax3, ax4])

           cbar = fig.colorbar(all_axes[0].collections[0], ax=all_axes, 
                  pad=0.02, fraction=0.06,aspect=40,extend='both',orientation='horizontal')

# Access the colorbar and set ticks/labels
           cbar = ax1.collections[0].colorbar
           cbar.set_ticks(tick_doys)  # Set ticks at 10-day intervals (DOY)
#           cbar.set_ticklabels(tick_labels)  # Set the tick labels as calendar dates (e.g., 01 Mar, 11 Mar, ...)
           cbar.set_label('Precipitation (JJAS)', fontsize=12, fontweight='bold')
           cbar.ax.tick_params(axis='x', rotation=0, labelsize=10) 
# Add arrows at both ends of the colorbar to indicate the range
#           cbar.ax.annotate('Start: 01 Mar', xy=(0, 0), xytext=(0, -1.5), ha='center', va='center', textcoords='offset points',
#                 arrowprops=dict(arrowstyle='->', lw=1.5))
#           cbar.ax.annotate('End: 31 Jul', xy=(1, 0), xytext=(0, -1.5), ha='center', va='center', textcoords='offset points',
#                 arrowprops=dict(arrowstyle='->', lw=1.5))

# Title and axis labels
    

    ax1.coastlines(resolution='10m',color='black',linewidth=1)
    ax1.set_title('IMERG (Reference)',fontweight='bold', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.xaxis.set_tick_params(labelsize=2)
    ax1.yaxis.set_tick_params(labelsize=2)
    ax1.set_aspect('auto')
    gls = ax1.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False

    ax2.coastlines(resolution='10m',color='black',linewidth=1)
    ax2.set_title('ICON (10KM)',fontweight='bold', fontsize=14)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.xaxis.set_tick_params(labelsize=2)
    ax2.yaxis.set_tick_params(labelsize=2)
    ax2.set_aspect('auto')
    gls = ax2.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False


    ax3.coastlines(resolution='10m',color='black',linewidth=1)
#    ax2.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax3.set_title('ICON (40KM)',fontweight='bold', fontsize=14)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.xaxis.set_tick_params(labelsize=2)
    ax3.yaxis.set_tick_params(labelsize=2)
    ax3.set_aspect('auto')
    gls = ax3.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False

    ax4.coastlines(resolution='10m',color='black',linewidth=1)
#    ax3.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax4.set_title('ICON (80KM)',fontweight='bold', fontsize=14)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.xaxis.set_tick_params(labelsize=2)
    ax4.yaxis.set_tick_params(labelsize=2)
    ax4.set_aspect('auto')
    gls = ax4.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False
#    _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)

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
#plt.savefig(plot_name)
fig.savefig("Fig6_Precipitation.pdf", bbox_inches='tight', pad_inches=0.1)
