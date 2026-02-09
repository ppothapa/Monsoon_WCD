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

data_base_dir_08 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/tot_prec'
model_data_08    =  'tot_prec_30_day.nc'


# Directoy for 40KM simulation

data_base_dir_06 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/tot_prec'
model_data_06    =  'tot_prec_30_day.nc'



# Directory for 80KM simulation

data_base_dir_05 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/tot_prec'
model_data_05    =  'tot_prec_30_day.nc'


# Observational Data Sets for IMERG

data_base_dir_obs_cmorph = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/IMERG/day_nc4_files/post_processed_files'
model_data_obs_cmorph    =  'precipitation_cdo_all_rmp_10years.nc'


# Observational Data Sets for CPC

data_base_dir_obs_cpc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CPC_data/daily'
model_data_obs_cpc    =  'precip.daily_rmp_10years.nc'


# Observational Data Sets for ERA5

data_base_dir_obs_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/daily'
model_data_obs_era5    =  'pr_day_reanalysis_era5_r1i1p1_daily_rmp_10years.nc'


# Obesrvational Data Set for MSWEP

data_base_dir_obs_mswep = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/MSWEP/Data_2006_2022'
model_data_obs_mswep    = 'precip_rmp_10years.nc'

# CMORPH Data ((Time step = 3653))

data_base_dir_obs_cmorph = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CMORPH/data/CMORPH/daily_nc'
model_data_obs_cmorph    = 'daily_rmp_10years.nc'



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
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_obs_cmorph    = os.path.join(data_base_dir_obs_cmorph, model_data_obs_cmorph)
model_data_dir_obs_cpc    = os.path.join(data_base_dir_obs_cpc, model_data_obs_cpc)
model_data_dir_obs_era5    = os.path.join(data_base_dir_obs_era5, model_data_obs_era5)
model_data_dir_obs_mswep    = os.path.join(data_base_dir_obs_mswep, model_data_obs_mswep)

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)  
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_cmorph = os.path.join(data_base_mask,mask_cmorph)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_cmorph  = xr.open_dataset(model_data_dir_obs_cmorph)
ds_obs_cpc  = xr.open_dataset(model_data_dir_obs_cpc)
ds_obs_era5  = xr.open_dataset(model_data_dir_obs_era5)
ds_obs_mswep  = xr.open_dataset(model_data_dir_obs_mswep)
###

# Calculating uniqe years in the data-sets.
years = np.unique(ds_obs_mswep['time'].dt.year.values)
print(years)

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

reg = 'South_Asia','North Africa','North America'

for x in reg:

    if x == 'South_Asia':
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
    var_08 = ds_08[plot_var_key]
    var_06 = ds_06[plot_var_key]
    var_05 = ds_05[plot_var_key] 
    var_obs_cmorph = ds_obs_cmorph['cmorph']
    var_obs_cpc = ds_obs_cpc['precip']
    var_obs_era5 = ds_obs_era5['pr']
    var_obs_mswep = ds_obs_mswep['precipitation']


# Masks
    mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
    mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
    mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
    mask_era5 = ds_mask_era5["__xarray_dataarray_variable__"]
    mask_cmorph = ds_mask_cmorph["__xarray_dataarray_variable__"]
    mask_cpc = ds_mask_cpc["__xarray_dataarray_variable__"]
    mask_mswep = ds_mask_mswep["__xarray_dataarray_variable__"]

## Selecting Lat Lon

    mask_08 = mask_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_06 = mask_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_05 = mask_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_cmorph = mask_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_cpc = mask_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_era5 = mask_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_mswep = mask_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))



# Select Lat & Lon focussing on the tropics 
    var_08 = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06 = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05 = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_cmorph = var_obs_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_cpc = var_obs_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_era5 = var_obs_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_mswep = var_obs_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select the Seasons to Caluculate the Monsoon Onset & Do Pentad Centered over a day!      
    var_08_summer = var_08.sel(time=var_08.time.dt.month.isin([3,4,5,6,7]))
    var_08_summer = var_08_summer.rolling(time=5, center=True).mean()

    var_06_summer = var_06.sel(time=var_06.time.dt.month.isin([3,4,5,6,7]))
    var_06_summer = var_06_summer.rolling(time=5, center=True).mean()

    var_05_summer = var_05.sel(time=var_05.time.dt.month.isin([3,4,5,6,7]))
    var_05_summer = var_05_summer.rolling(time=5, center=True).mean()   
 
    var_obs_cmorph_summer = var_obs_cmorph.sel(time=var_obs_cmorph.time.dt.month.isin([3,4,5,6,7]))
    var_obs_cmorph_summer = var_obs_cmorph_summer.rolling(time=5, center=True).mean()

    var_obs_cpc_summer = var_obs_cpc.sel(time=var_obs_cpc.time.dt.month.isin([3,4,5,6,7]))
    var_obs_cpc_summer = var_obs_cpc_summer.rolling(time=5, center=True).mean()

    var_obs_era5_summer = var_obs_era5.sel(time=var_obs_era5.time.dt.month.isin([3,4,5,6,7]))
    var_obs_era5_summer = var_obs_era5_summer.rolling(time=5, center=True).mean()

    var_obs_mswep_summer = var_obs_mswep.sel(time=var_obs_mswep.time.dt.month.isin([3,4,5,6,7]))
    var_obs_mswep_summer = var_obs_mswep_summer.rolling(time=5, center=True).mean()


#   Multiply with the Mask. 
    var_08_summer            = var_08_summer * mask_cmorph 
    var_06_summer            = var_06_summer * mask_cmorph
    var_05_summer            = var_05_summer * mask_cmorph
    var_obs_cmorph_summer     = var_obs_cmorph_summer  * mask_cmorph
    var_obs_cpc_summer       = var_obs_cpc_summer  * mask_cpc
    var_obs_era5_summer      = var_obs_era5_summer * mask_era5
    var_obs_mswep_summer     = var_obs_mswep_summer * mask_mswep

    # Define the monsoon onset threshold (5mm/day) and the minimum number of consecutive days (3 days)
    threshold = 5  # mm/day
    min_consecutive_days = 3
   
    onset_dates = []
    onset_data = {}    

    model_data = {
    'var_08_summer': var_08_summer,      # Already loaded xarray
    'var_06_summer': var_06_summer,
    'var_05_summer': var_05_summer,
    'var_obs_cmorph_summer': var_obs_cmorph_summer,
    'var_obs_cpc_summer': var_obs_cpc_summer,
    'var_obs_era5_summer': var_obs_era5_summer,
    'var_obs_mswep_summer': var_obs_mswep_summer,
     } 

    model_names=['var_08_summer', 'var_06_summer', 'var_05_summer', 'var_obs_cmorph_summer' , 'var_obs_cpc_summer', 'var_obs_era5_summer',  'var_obs_mswep_summer']

    for model_name in model_names:
        for year in years:
            print(year)
            print(model_name)
            ds = model_data[model_name]
            print(ds.shape)
            year_data = ds.sel(time=ds.time.dt.year == year)
            condition_met = (year_data > threshold)
            rolling_sum = condition_met.rolling(time=min_consecutive_days).sum()
            onset_flag = rolling_sum >= min_consecutive_days


    # Get first occurrence of onset per grid point
            onset_idx = onset_flag.argmax(dim='time')
            onset_date = year_data.time[onset_idx]


    # Avoid false positives where onset never happened
            valid_onset = onset_flag.any(dim='time')
            onset_date = onset_date.where(valid_onset)

            onset_dates.append(onset_date)
            print(onset_date.shape)
# Combine all years into a single DataArray
        onset_dates_all_years = xr.concat(onset_dates, dim='year')
        print(onset_dates_all_years.shape)
        onset_dates_all_years['year'] = years

# Convert to day of year
        onset_doy = onset_dates_all_years.dt.dayofyear

# Average across years
        mean_onset_doy = onset_doy.mean(dim='year')
        key = f"{model_name}_onset"
        onset_data[key] = mean_onset_doy
# Clear the onset_dates list to avoid accumulation in the next iteration
        onset_dates = []


# v vonset_data[key] = mean_onset_doy

    var_08_onset        = onset_data["var_08_summer_onset"]
    var_06_onset        = onset_data["var_06_summer_onset"]
    var_05_onset        = onset_data["var_05_summer_onset"]
    var_obs_cmorph_onset = onset_data["var_obs_cmorph_summer_onset"]  
    var_obs_cpc_onset   = onset_data["var_obs_cpc_summer_onset"] 
    var_obs_era5_onset  = onset_data["var_obs_era5_summer_onset"]
    var_obs_mswep_onset = onset_data["var_obs_mswep_summer_onset"]

    def smart_clean_onset(onset_da, max_reasonable_doy=300):
        """
        Smart cleaning for monsoon onset data
        - Replaces NaN with 'no onset' indicator
        - Handles unrealistic values
        """
    # Replace NaN with a value indicating 'no onset' (beyond monsoon season)
        no_onset_value = 300  # July 19th - well beyond normal monsoon onset
    
        cleaned = onset_da.fillna(no_onset_value)
    
    # Also cap unrealistic large values
        cleaned = cleaned.where(cleaned <= max_reasonable_doy, no_onset_value)
    
    # Handle infinite values
        cleaned = cleaned.where(np.isfinite(cleaned), no_onset_value)
    
        return cleaned

    var_08_onset_clean = smart_clean_onset(var_08_onset)
    var_06_onset_clean = smart_clean_onset(var_06_onset)
    var_05_onset_clean = smart_clean_onset(var_05_onset)
    var_obs_cmorph_onset_clean = smart_clean_onset(var_obs_cmorph_onset)
    var_obs_mswep_onset_clean = smart_clean_onset(var_obs_mswep_onset)    

# Difference in the domain
    var_08_onset_difference        = var_08_onset_clean - var_obs_cmorph_onset_clean 
    var_06_onset_difference        = var_06_onset_clean - var_obs_cmorph_onset_clean
    var_05_onset_difference        = var_05_onset_clean - var_obs_cmorph_onset_clean
 
    print(f"the shape of var_08_onset_difference is {var_08_onset_difference.shape}")

    cmap = mpl.colormaps['Spectral_r']
    cmap = mpl.colormaps['bwr']
    # Define the boundaries for the colorbar ticks (DOY from March 1st to July 31st)
    tick_doys = np.arange(-60, 61, 2)  # 10-day intervals within the March-July range
    norm = BoundaryNorm(tick_doys, cmap.N)

#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
#    proj = ccrs.PlateCarree()
   
    all_axes = [] 
    if i == 1:
             i = 1

    elif i == 2: 
             i = 5

    elif i == 3: 
             i = 9

#    ax1 = fig.add_subplot(3,4,i, projection=proj)
#    var_obs_cmorph_onset.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)

    ax1 = fig.add_subplot(3,4,i, projection=proj)
     # Plot countours for 10KM, 40KM and 80KM
    var_08_onset_difference.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # FORCE EXTENT

#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5) 
    
    ax2 = fig.add_subplot(3,4,i+1, projection=proj) 
    var_06_onset_difference.plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # FORCE EXTENT
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)
 
    ax3 = fig.add_subplot(3,4,i+2, projection=proj)
    var_05_onset_difference.plot.contourf(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,  add_colorbar=False, add_labels=False)
    ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # FORCE EXTENT

#     plot = diff_sum_win_05.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['b'],linewidths=2.5)

    
    ax1.coastlines(resolution='10m',color='black',linewidth=1)
    ax1.set_title('ICON (10KM)',fontweight='bold', fontsize=10)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.xaxis.set_tick_params(labelsize=2)
    ax1.yaxis.set_tick_params(labelsize=2)
    ax1.set_aspect('auto')
    gls = ax1.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False


    ax2.coastlines(resolution='10m',color='black',linewidth=1)
#    ax2.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax2.set_title('ICON (40KM)',fontweight='bold', fontsize=10)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.xaxis.set_tick_params(labelsize=2)
    ax2.yaxis.set_tick_params(labelsize=2)
    ax2.set_aspect('auto')
    gls = ax2.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False

    ax3.coastlines(resolution='10m',color='black',linewidth=1)
#    ax3.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    ax3.set_title('ICON (80KM)',fontweight='bold', fontsize=10)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.xaxis.set_tick_params(labelsize=2)
    ax3.yaxis.set_tick_params(labelsize=2)
    ax3.set_aspect('auto')
    gls = ax3.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False
#    _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)


#    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
#    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
#    ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


#    if i == 6 or i == 2 or i ==4:
#	     			var_obs_mswep_onset.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, cbar_kwargs={'pad': 0.02})
#
#    else:
#         var_obs_mswep_onset.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, cbar_kwargs={'pad': 0.02})
    
    # Replace the fourth panel with a box plot
    ax4 = fig.add_subplot(3,4,i+3)  # No projection for box plot

   # Prepare data for box plot
    diff_data = [
    var_08_onset_difference.values.flatten(),
    var_06_onset_difference.values.flatten(), 
    var_05_onset_difference.values.flatten()
     ]

   # Remove NaN values
    diff_data_clean = [d[~np.isnan(d) & (d != 0)] for d in diff_data]

    # Create box plot
    COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
    violin_parts = ax4.violinplot(diff_data_clean, showmeans=True, showmedians=True)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['08km', '06km', '05km'])
    ax4.set_ylabel('Onset Difference (days)')
    ax4.set_title('Distribution of Onset Differences')
    ax4.grid(True, alpha=0.3)

# Apply your color scheme
    for pc, color in zip(violin_parts['bodies'], COLORS):
     pc.set_facecolor(color)
     pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmeans'].set_color('brown')
    violin_parts['cmeans'].set_linewidth(2)




# You might want to set consistent y-axis limits
    all_diffs = np.concatenate(diff_data_clean)
    ax4.set_ylim(np.nanmin(all_diffs) - 5, np.nanmax(all_diffs) + 5)

   
    if i == 9: 
           all_axes.extend([ax1, ax2, ax3, ax4])

           cbar = fig.colorbar(all_axes[0].collections[0], ax=all_axes, 
                  pad=0.02, fraction=0.06,aspect=40,extend='both',orientation='horizontal')

# Add coast line 
    # Set colorbar with calendar dates for March 1st to July 31st (DOY 60 to 210)
#           tick_labels = pd.to_datetime(tick_doys, origin='2000-01-01', unit='D').strftime('%d %b')  # Convert DOY to calendar dates

# Access the colorbar and set ticks/labels
           cbar = ax1.collections[0].colorbar
           cbar_ticks = np.arange(-60, 61, 10)
           cbar.set_ticks(cbar_ticks)
#           cbar.set_ticks(tick_doys)  # Set ticks at 10-day intervals (DOY)
#           cbar.set_ticklabels(tick_labels)  # Set the tick labels as calendar dates (e.g., 01 Mar, 11 Mar, ...)
           cbar.set_label('Average Monsoon Onset Bias')
           cbar.ax.tick_params(axis='x', rotation=45) 
# Add arrows at both ends of the colorbar to indicate the range
#           cbar.ax.annotate('Start: 01 Mar', xy=(0, 0), xytext=(0, -1.5), ha='center', va='center', textcoords='offset points',
#                 arrowprops=dict(arrowstyle='->', lw=1.5))
#           cbar.ax.annotate('End: 31 Jul', xy=(1, 0), xytext=(0, -1.5), ha='center', va='center', textcoords='offset points',
#                 arrowprops=dict(arrowstyle='->', lw=1.5))

# Title and axis labels
    

#plt.tight_layout()
plt.savefig(plot_name)
fig.savefig("Fig3_Monsoon_Onset_Bias_CMORPH.pdf", bbox_inches='tight', pad_inches=0.1)
