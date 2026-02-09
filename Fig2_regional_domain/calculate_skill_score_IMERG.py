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

##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

## Function of Skill Scores. 


def compute_skill_scores(model_mask, obs_mask):
    # Flatten to avoid dimension mismatch
    model = model_mask.values.flatten()
    obs = obs_mask.values.flatten()

    H = np.sum((model == 1) & (obs == 1))  # Hits
    M = np.sum((model == 0) & (obs == 1))  # Misses
    F = np.sum((model == 1) & (obs == 0))  # False Alarms
    C = np.sum((model == 0) & (obs == 0))  # Correct Negatives
    total = len(model)

    hit_rate = H / (H + M) if (H + M) > 0 else np.nan
    false_alarm_rate = F / (F + H) if (F + H) > 0 else np.nan
    bias_score = (H + F) / (H + M) if (H + M) > 0 else np.nan
    accuracy = (H + C) / total if total > 0 else np.nan
    csi = H / (H + M + F) if (H + M + F) > 0 else np.nan

    return {
        'Hit Rate': hit_rate,
        'False Alarm Rate': false_alarm_rate,
        'Bias Score': bias_score,
        'Accuracy': accuracy,
        'CSI': csi
    }






Region="Australia"

if Region == 'South_Asia':
    # Select a Region
    #South Asia
    lat_min=0
    lat_max=50
    lon_min=60
    lon_max=150
    plot_name='South_Asia'

elif Region == 'Africa':
    lat_min=-40
    lat_max=20
    lon_min=-65
    lon_max=65
    plot_name='Africa'

elif Region == 'America':
    lat_min=-40
    lat_max=30
    lon_min=-120
    lon_max=0
    plot_name='America'

elif Region == 'Australia':
    lat_min=-40
    lat_max=0
    lon_min=90
    lon_max=180
    plot_name='Australia'



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

data_base_dir_obs_imerg = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/IMERG/day_nc4_files/post_processed_files'
model_data_obs_imerg    =  'precipitation_cdo_all_rmp_10years.nc'


# Observational Data Sets for CPC

data_base_dir_obs_cpc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CPC_data/daily'
model_data_obs_cpc    =  'precip.daily_rmp_10years.nc'


# Observational Data Sets for ERA5

data_base_dir_obs_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/daily'
model_data_obs_era5    =  'pr_day_reanalysis_era5_r1i1p1_daily_rmp_10years.nc'


# Obesrvational Data Set for MSWEP

data_base_dir_obs_mswep = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/MSWEP/Data_2006_2022'
model_data_obs_mswep    = 'precip_rmp_10years.nc'



# Mask Data Sets

# Mask Data Sets
data_base_mask = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig1_domains/original_masks'
mask_08='08_mask_2mm_55.nc'
mask_06='06_mask_2mm_55.nc'
mask_05='05_mask_2mm_55.nc'
mask_era5='era5_mask_2mm_55.nc'
mask_imerg='imerg_mask_2mm_55.nc'
mask_cpc='cpc_mask_2mm_55.nc'
mask_mswep='mswep_mask_2mm_55.nc'


# netCDF variable name of variable to plot
var='tot_prec','t'

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_imerg = os.path.join(data_base_mask,mask_imerg)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)


ds_mask_08 = xr.open_dataset(data_dir_mask_08)
ds_mask_06 = xr.open_dataset(data_dir_mask_06)
ds_mask_05 = xr.open_dataset(data_dir_mask_05)
ds_mask_imerg = xr.open_dataset(data_dir_mask_imerg)
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


fig = plt.figure(figsize=(12, 12), constrained_layout=True)

colors = ['r', 'g', 'b', 'k']
labels = ['ICON-10KM', 'ICON-40KM', 'ICON-80KM', 'OBS (MSWEP)']
Titles = ['Asian Summer Monsoon', 'North African Monsoon', 'South African Monsoon', 'North American Monsoon', 'South American Monsoon','Australian Monsoon']

reg = 'South_Asia','North Africa','South Africa','North America', 'South America','Australia'

scores_08 = {};scores_06 = {};scores_05 = {}

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
        lat_min=-2
        lat_max=20
        lon_min=-65
        lon_max=65
        plot_name='North Africa'
        i=2

    elif x == 'South Africa':
        lat_min=-40
        lat_max=0
        lon_min=-5
        lon_max=85
        plot_name='South Africa'
        i=3

    elif x == 'North America':
        lat_min=0
        lat_max=30
        lon_min=-120
        lon_max=-75
        plot_name='North America'
        i=4

    elif x == 'South America':
        lat_min=-40
        lat_max=5
        lon_min=-90
        lon_max=-30
        plot_name='South America'
        i=5

    elif x == 'Australia':
        lat_min=-40
        lat_max=0
        lon_min=90
        lon_max=180
        plot_name='Australia'
        i=6
    
    # Plot Name and Name of the Variable for Plotting.
    plot_var_key = 'tot_prec'
    print(plot_var_key)
    plot_name = 'all_' + 'Mask'
    print(plot_name)

    mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
    mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
    mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
    mask_era5 = ds_mask_era5["__xarray_dataarray_variable__"]
    mask_imerg = ds_mask_imerg["__xarray_dataarray_variable__"]
    mask_cpc = ds_mask_cpc["__xarray_dataarray_variable__"]
    mask_mswep = ds_mask_mswep["__xarray_dataarray_variable__"]



    # Select Lat & Lon focussing on the tropics
    mask_08 = mask_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_06 = mask_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_05 = mask_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_imerg = mask_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_cpc = mask_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_era5 = mask_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    mask_mswep = mask_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

# IMERG DATA 
   
#   Calclate the Skill scores Here.     
#    scores_08 = {};scores_06 = {};scores_05 = {}    

    scores_08[x] = compute_skill_scores(mask_08, mask_imerg)
    scores_06[x] = compute_skill_scores(mask_06, mask_imerg)
    scores_05[x] = compute_skill_scores(mask_05, mask_imerg)

    with open('scores_08.txt', 'w') as f:
    	for region, metrics in scores_08.items():
        	f.write(f"Region: {region}\n")
        	for key, value in metrics.items():
            		f.write(f"  {key}: {value:.4f}\n")
        	f.write("\n")

    with open('scores_06.txt', 'w') as f:
        for region, metrics in scores_06.items():
                f.write(f"Region: {region}\n")
                for key, value in metrics.items():
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")

    with open('scores_05.txt', 'w') as f:
        for region, metrics in scores_05.items():
                f.write(f"Region: {region}\n")
                for key, value in metrics.items():
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")


    ## Save skill scores to a text file
    #with open('scores_08.txt', 'w') as f:
   # 	for key, value in scores_08.items():
   #     	f.write(f"{key}: {value:.4f}\n")


    


#    var_obs_imerg = var_obs_imerg.transpose()


#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
    ax = fig.add_subplot(3,2,i, projection=ccrs.PlateCarree())

    if i == 6 or i == 2 or i ==4:
	     			mask_imerg.plot(ax=ax,transform=ccrs.PlateCarree(), cmap='Greys',vmin=0,vmax=1,cbar_kwargs={'label': 'Mask', 'extend': 'both',  'shrink': .8, 'orientation': 'vertical'},add_labels=False)

    else:
        mask_imerg.plot(ax=ax,transform=ccrs.PlateCarree(), cmap='Greys',vmin=0,vmax=1,add_colorbar=False,add_labels=False)


# Add coast line 
    ax.coastlines(resolution='10m',color='brown',linewidth=1)
    ax.set_title(f'ICON Simulations Monsoon Domains')
    ax.set_title(Titles[i-1],fontweight="bold")
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
#    ax.set_yticklabels(ax.get_yticks(), weight='bold')
#    ax.set_xticklabels(ax.get_xticks(), rotation=0, weight='bold')

#    for label in ax.get_yticklabels():
#    				     label.set_weight('bold')

#    yticks = ax.get_yticks()
#    ax.set_yticks(yticks)  # explicitly fix the ticks
#    ax.set_yticklabels(yticks, weight='bold')  # now this is safe

#    xticks = ax.get_xticks()
#    ax.set_xticks(xticks)  # explicitly fix the ticks
#    ax.set_xticklabels(xticks, weight='bold')  # now this is safe

    ax.set_xlabel("Longitude", fontsize=8, weight='bold')
    ax.set_ylabel("Latitude", fontsize=8, weight='bold')


#    ax.set_aspect(1.5)
    ax.set_aspect('auto')
#    cbar_ax.yaxis.label.set_size(20)
#     ax1.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
    gls = ax.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False
#    _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)

#plt.tight_layout()
plt.savefig(plot_name)
