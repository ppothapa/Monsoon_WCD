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

import netCDF4

##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

# base directory where your analysis data is stobrown GPU
# Directory of 10KM simulation 


data_base_dir = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig4_Monsoon_Bias/data_generation'

# CMORPH
model_data_08_cmorph    =  'diff_sum_08_obs_cmorph.nc'

model_data_06_cmorph    =  'diff_sum_06_obs_cmorph.nc'

model_data_05_cmorph    =  'diff_sum_05_obs_cmorph.nc'

model_data_obs_cmorph    =  'cli_cmorph_summer.nc'


#MSWEP
model_data_08_mswep    =  'diff_sum_08_obs_mswep.nc'

model_data_06_mswep    =  'diff_sum_06_obs_mswep.nc'

model_data_05_mswep    =  'diff_sum_05_obs_mswep.nc'

model_data_obs_mswep    =  'cli_mswep_summer.nc'


#ERA5
model_data_08_era5    =  'diff_sum_08_obs_era5.nc'

model_data_06_era5    =  'diff_sum_06_obs_era5.nc'

model_data_05_era5    =  'diff_sum_05_obs_era5.nc'

model_data_obs_era5    =  'cli_era5_summer.nc'


#CPC
model_data_08_cpc    =  'diff_sum_08_obs_cpc.nc'

model_data_06_cpc    =  'diff_sum_06_obs_cpc.nc'

model_data_05_cpc    =  'diff_sum_05_obs_cpc.nc'

model_data_obs_cpc    =  'cli_cpc_summer.nc'


# Mask Data Sets
data_base_mask = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig1_domains/original_masks'
mask_08='08_mask_2mm_55_na.nc'
mask_06='06_mask_2mm_55_na.nc'
mask_05='05_mask_2mm_55_na.nc'
mask_era5='era5_mask_2mm_55_na.nc'
mask_cmorph='cmorph_mask_2mm_55_na.nc'
mask_cpc='cpc_mask_2mm_55.nc'
mask_mswep='mswep_mask_2mm_55_na.nc'


# netCDF variable name of variable to plot
var='tot_prec','t'

# Read the Model Data from Here
model_data_dir_08_cmorph    = os.path.join(data_base_dir, model_data_08_cmorph)
model_data_dir_06_cmorph    = os.path.join(data_base_dir, model_data_06_cmorph)
model_data_dir_05_cmorph    = os.path.join(data_base_dir, model_data_05_cmorph)

model_data_dir_08_mswep    = os.path.join(data_base_dir, model_data_08_mswep)
model_data_dir_06_mswep    = os.path.join(data_base_dir, model_data_06_mswep)
model_data_dir_05_mswep    = os.path.join(data_base_dir, model_data_05_mswep)


model_data_dir_08_era5    = os.path.join(data_base_dir, model_data_08_era5)
model_data_dir_06_era5    = os.path.join(data_base_dir, model_data_06_era5)
model_data_dir_05_era5    = os.path.join(data_base_dir, model_data_05_era5)


model_data_dir_08_cpc    = os.path.join(data_base_dir, model_data_08_cpc)
model_data_dir_06_cpc    = os.path.join(data_base_dir, model_data_06_cpc)
model_data_dir_05_cpc    = os.path.join(data_base_dir, model_data_05_cpc)
##

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)  
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_cmorph = os.path.join(data_base_mask,mask_cmorph)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)


#Read the OpenDataset. 
ds_08_cmorph  = xr.open_dataset(model_data_dir_08_cmorph)
ds_06_cmorph  = xr.open_dataset(model_data_dir_06_cmorph)
ds_05_cmorph  = xr.open_dataset(model_data_dir_05_cmorph)


ds_08_mswep  = xr.open_dataset(model_data_dir_08_mswep)
ds_06_mswep  = xr.open_dataset(model_data_dir_06_mswep)
ds_05_mswep  = xr.open_dataset(model_data_dir_05_mswep)


ds_08_era5  = xr.open_dataset(model_data_dir_08_era5)
ds_06_era5  = xr.open_dataset(model_data_dir_06_era5)
ds_05_era5  = xr.open_dataset(model_data_dir_05_era5)


ds_08_cpc  = xr.open_dataset(model_data_dir_08_cpc)
ds_06_cpc  = xr.open_dataset(model_data_dir_06_cpc)
ds_05_cpc  = xr.open_dataset(model_data_dir_05_cpc)

###

##
ds_mask_08 = xr.open_dataset(data_dir_mask_08)
ds_mask_06 = xr.open_dataset(data_dir_mask_06)
ds_mask_05 = xr.open_dataset(data_dir_mask_05)
ds_mask_cmorph = xr.open_dataset(data_dir_mask_cmorph)
ds_mask_cpc = xr.open_dataset(data_dir_mask_cpc)
ds_mask_era5 = xr.open_dataset(data_dir_mask_era5)
ds_mask_mswep = xr.open_dataset(data_dir_mask_mswep)



# Color Scales Range
a = [0,.01,.1,.25,.5,1,1.5,2,3,4,6,8,10,15,20,30]
a = [0,.005,.01,.15,.25,0.5,1,2,3,4,5,6,7,8,20,30]
a = [-30,-20,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,20,30]
bounds = np.linspace(-8,8,10)
bounds = np.arange(1,10.1,1)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)


# The plotting call from here 

fig = plt.figure(figsize=(12, 10), constrained_layout=True)

colors = ['r', 'g', 'b', 'k']
labels = ['ICON-10KM', 'ICON-40KM', 'ICON-80KM', 'OBS (MSWEP)']
Titles = ['Asian Summer Monsoon', 'North African Monsoon', 'North American Monsoon']

reg = 'South Asia','North Africa','North America'

for x in reg:

    if x == 'South Asia':
    # Select a Region
    #South Asia
        lat_min=0
        lat_max=40
        lon_min=60
        lon_max=120
        plot_name='South_Asia'
        i=1

    elif x == 'North Africa':
        lat_min=-2
        lat_max=20
        lon_min=-65
        lon_max=65
        plot_name='North Africa'
        i=2

    elif x == 'North America':
        lat_min=0
        lat_max=30
        lon_min=-120
        lon_max=-75
        plot_name='North America'
        i=3
    
    # Plot Name and Name of the Variable for Plotting.
    plot_var_key = 'tot_prec'
    print(plot_var_key)
    plot_name = 'all_' + 'Monsoon_Onset'
    print(plot_name)

# Read the Variable from Here.  
    var_08_cmorph = ds_08_cmorph["__xarray_dataarray_variable__"]
    var_06_cmorph = ds_06_cmorph["__xarray_dataarray_variable__"]
    var_05_cmorph = ds_05_cmorph["__xarray_dataarray_variable__"] 
##  
    var_08_mswep = ds_08_mswep["__xarray_dataarray_variable__"]
    var_06_mswep = ds_06_mswep["__xarray_dataarray_variable__"]
    var_05_mswep = ds_05_mswep["__xarray_dataarray_variable__"]

##  
    var_08_era5 = ds_08_era5["__xarray_dataarray_variable__"]
    var_06_era5 = ds_06_era5["__xarray_dataarray_variable__"]
    var_05_era5 = ds_05_era5["__xarray_dataarray_variable__"]

###
    var_08_cpc = ds_08_cpc["__xarray_dataarray_variable__"]
    var_06_cpc = ds_06_cpc["__xarray_dataarray_variable__"]
    var_05_cpc = ds_05_cpc["__xarray_dataarray_variable__"]

# Masks
    mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
    mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
    mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
    mask_era5 = ds_mask_era5["__xarray_dataarray_variable__"]
    mask_cmorph = ds_mask_cmorph["__xarray_dataarray_variable__"]
    mask_cpc = ds_mask_cpc["__xarray_dataarray_variable__"]
    mask_mswep = ds_mask_mswep["__xarray_dataarray_variable__"]

## Selecting Lat Lon

    mask_cmorph = mask_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_mswep = mask_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_era5 = mask_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_cpc = mask_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select Lat & Lon focussing on the tropics 
    var_08_cmorph = var_08_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_cmorph = var_06_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_cmorph = var_05_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

    var_08_mswep = var_08_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_mswep = var_06_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_mswep = var_05_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

    var_08_era5 = var_08_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_era5 = var_06_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_era5 = var_05_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

    var_08_cpc = var_08_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_cpc = var_06_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_cpc = var_05_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


#   Multiply with the Mask. 
    var_08_cmorph            = var_08_cmorph * mask_cmorph 
    var_06_cmorph            = var_06_cmorph * mask_cmorph
    var_05_cmorph            = var_05_cmorph * mask_cmorph

    var_08_mswep            = var_08_mswep * mask_mswep
    var_06_mswep            = var_06_mswep * mask_mswep
    var_05_mswep            = var_05_mswep * mask_mswep

    var_08_era5            = var_08_era5 * mask_era5
    var_06_era5            = var_06_era5 * mask_era5
    var_05_era5            = var_05_era5 * mask_era5

    var_08_cpc            = var_08_cpc * mask_cpc
    var_06_cpc            = var_06_cpc * mask_cpc
    var_05_cpc            = var_05_cpc * mask_cpc

    tick_doys = np.arange(-8, 8, 1)  # 10-day intervals within the March-July range

#  Preparing for the figures & Spceifying the Projections
#    proj = ccrs.PlateCarree()
   
    all_axes = [] 
    if i == 1:
             i = 1

    elif i == 2: 
             i = 5

    elif i == 3: 
             i = 9

    # Replace the fourth panel with a box plot
    ax1 = fig.add_subplot(3,4,i)  # No projection for box plot

   # Prepare data for box plot
    diff_data = [
    var_08_cmorph.values.flatten(),
    var_06_cmorph.values.flatten(), 
    var_05_cmorph.values.flatten()
     ]

   # Remove NaN values
    diff_data_clean = [d[~np.isnan(d) ] for d in diff_data]

    # Create box plot
    COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
    violin_parts = ax1.violinplot(diff_data_clean, showmeans=True, showmedians=True,  showextrema=False)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['10km', '40km', '80km'])
    ax1.set_ylabel('Onset Difference (days)')
    ax1.set_title('ICON - CMORPH')
    ax1.grid(True, alpha=0.3)

# You might want to set consistent y-axis limits
    all_diffs = np.concatenate(diff_data_clean)
    #ax1.set_ylim(np.nanmin(all_diffs) - 5, np.nanmax(all_diffs) + 5)
    ax1.set_ylim(-8,8)
    # Apply your color scheme
    for pc, color in zip(violin_parts['bodies'], COLORS):
     pc.set_facecolor(color)
     pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmeans'].set_color('brown')
    violin_parts['cmeans'].set_linewidth(2)



    ax1.text(0.02, 0.98, x, transform=ax1.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


##   Here starts ax2 

    ax2 = fig.add_subplot(3,4,i+1)  # No projection for box plot

   # Prepare data for box plot
    diff_data = [
    var_08_mswep.values.flatten(),
    var_06_mswep.values.flatten(),
    var_05_mswep.values.flatten()
     ]

   # Remove NaN values
    diff_data_clean = [d[~np.isnan(d) ] for d in diff_data]

    # Create box plot
    COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
    violin_parts = ax2.violinplot(diff_data_clean, showmeans=True, showmedians=True,  showextrema=False)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['10km', '40km', '80km'])
    ax2.set_ylabel('Onset Difference (days)')
    ax2.set_title('ICON - MSWEP')
    ax2.grid(True, alpha=0.3)

# You might want to set consistent y-axis limits
    all_diffs = np.concatenate(diff_data_clean)
    #ax2.set_ylim(np.nanmin(all_diffs) - 5, np.nanmax(all_diffs) + 5)
    ax2.set_ylim(-8,8)
    # Apply your color scheme
    for pc, color in zip(violin_parts['bodies'], COLORS):
     pc.set_facecolor(color)
     pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmeans'].set_color('brown')
    violin_parts['cmeans'].set_linewidth(2)
    


    ax2.text(0.02, 0.98, x, transform=ax2.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))



##  Here starts ax3

    ax3 = fig.add_subplot(3,4,i+2)  # No projection for box plot

   # Prepare data for box plot
    diff_data = [
    var_08_era5.values.flatten(),
    var_06_era5.values.flatten(),
    var_05_era5.values.flatten()
     ]

   # Remove NaN values
    diff_data_clean = [d[~np.isnan(d) ] for d in diff_data]

    # Create box plot
    COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
    violin_parts = ax3.violinplot(diff_data_clean, showmeans=True, showmedians=True, showextrema=False)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['10km', '40km', '80km'])
    ax3.set_ylabel('Onset Difference (days)')
    ax3.set_title('ICON - ERA5')
    ax3.grid(True, alpha=0.3)

# You might want to set consistent y-axis limits
    all_diffs = np.concatenate(diff_data_clean)
   # ax3.set_ylim(np.nanmin(all_diffs) - 5, np.nanmax(all_diffs) + 5)
    ax3.set_ylim(-8,8)

    # Apply your color scheme
    for pc, color in zip(violin_parts['bodies'], COLORS):
     pc.set_facecolor(color)
     pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmeans'].set_color('brown')
    violin_parts['cmeans'].set_linewidth(2)
    


    ax3.text(0.02, 0.98, x, transform=ax3.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


#  Here starts ax4

    ax4 = fig.add_subplot(3,4,i+3)  # No projection for box plot

   # Prepare data for box plot
    diff_data = [
    var_08_cpc.values.flatten(),
    var_06_cpc.values.flatten(),
    var_05_cpc.values.flatten()
     ]

   # Remove NaN values
    diff_data_clean = [d[~np.isnan(d)] for d in diff_data]

    # Create box plot
    COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
    violin_parts = ax4.violinplot(diff_data_clean, showmeans=True, showmedians=True, showextrema=False)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['10km', '40km', '80km'])
    ax4.set_ylabel('Onset Difference (days)')
    ax4.set_title('ICON - CPC')
    ax4.grid(True, alpha=0.3)

# You might want to set consistent y-axis limits
    all_diffs = np.concatenate(diff_data_clean)
    ax4.set_ylim(-4,4)

# Apply your color scheme
    for pc, color in zip(violin_parts['bodies'], COLORS):
     pc.set_facecolor(color)
     pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmeans'].set_color('brown')
    violin_parts['cmeans'].set_linewidth(2)


    ax4.text(0.02, 0.98, x, transform=ax4.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#plt.tight_layout()
plt.savefig(plot_name)
fig.savefig("all_violin_Precip_Bias.pdf", bbox_inches='tight', pad_inches=0.1)
