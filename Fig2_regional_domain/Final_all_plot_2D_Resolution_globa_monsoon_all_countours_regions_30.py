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

from scipy import ndimage

##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

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

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_obs_imerg    = os.path.join(data_base_dir_obs_imerg, model_data_obs_imerg)
model_data_dir_obs_cpc    = os.path.join(data_base_dir_obs_cpc, model_data_obs_cpc)
model_data_dir_obs_era5    = os.path.join(data_base_dir_obs_era5, model_data_obs_era5)
model_data_dir_obs_mswep    = os.path.join(data_base_dir_obs_mswep, model_data_obs_mswep)

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)  
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_imerg = os.path.join(data_base_mask,mask_imerg)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_imerg  = xr.open_dataset(model_data_dir_obs_imerg)
ds_obs_cpc  = xr.open_dataset(model_data_dir_obs_cpc)
ds_obs_era5  = xr.open_dataset(model_data_dir_obs_era5)
ds_obs_mswep  = xr.open_dataset(model_data_dir_obs_mswep)

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


# Normalize the bin between 0 and 1 (uneven bins are important here)
#norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]

# Color tuple for every bin
#C = np.array([[255,255,255],
#              [199,233,192],
#              [161,217,155],
#              [116,196,118],
#              [49,163,83],
#              [0,109,44],
#              [255,250,138],
#              [255,204,79],
#              [254,141,60],
#              [252,78,42],
#              [214,26,28],
#              [173,0,38],
#              [112,0,38],
#              [59,0,48],
#              [76,0,115],
#              [255,219,255]])

# Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
#COLORS = []
#for i, n in enumerate(norm):
#    COLORS.append((n, np.array(C[i])/255.))
#
## Create the colormap
#cmap = colors.LinearSegmentedColormap.from_list("precipitation", COLORS)
#
# The plotting call from here 


fig = plt.figure(figsize=(12, 12), constrained_layout=True)
#fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)
#fig, ax = plt.subplots(3, 2, figsize=(12, 12),layout='constrained')

colors = ['r', 'g', 'b', 'k']
labels = ['ICON (10KM)', 'ICON (40KM)', 'ICON (80KM)', 'OBS (IMERG)']
Titles = ['Asian Summer Monsoon', 'North African Monsoon', 'South African Monsoon', 'North American Monsoon', 'South American Monsoon','Australian Monsoon']

Titles = ['(a) SAsiaM/EAsiaM', '(b) WAfriM', '(c) SAfriM', '(d) NAmerM', '(e) SAmerM','(f) AusMCM']
reg = 'South_Asia','North Africa','South Africa','North America', 'South America','Australia'

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
        lat_max=35
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
    plot_name = 'all_' + 'contours'
    print(plot_name)

# Read the Variable from Here.  
    var_08 = ds_08[plot_var_key]
    var_06 = ds_06[plot_var_key]
    var_05 = ds_05[plot_var_key] 
    var_obs_imerg = ds_obs_imerg['precipitation']
    var_obs_cpc = ds_obs_cpc['precip']
    var_obs_era5 = ds_obs_era5['pr']
    var_obs_mswep = ds_obs_mswep['precipitation']

    mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
    mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
    mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
    mask_era5 = ds_mask_era5["__xarray_dataarray_variable__"]
    mask_imerg = ds_mask_imerg["__xarray_dataarray_variable__"]
    mask_cpc = ds_mask_cpc["__xarray_dataarray_variable__"]
    mask_mswep = ds_mask_mswep["__xarray_dataarray_variable__"]


# IMERG DATA 
    
    var_obs_imerg = var_obs_imerg.transpose()


# Select Lat & Lon focussing on the tropics 
    var_08 = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06 = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05 = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_imerg = var_obs_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_cpc = var_obs_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_era5 = var_obs_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_mswep = var_obs_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select the Seasons and Mean over the Seasons.      
    var_08_summer = var_08.sel(time=var_08.time.dt.month.isin([6, 7, 8, 9]))
    var_08_winter = var_08.sel(time=var_08.time.dt.month.isin([12, 1, 2, 3]))

    var_06_summer = var_06.sel(time=var_06.time.dt.month.isin([6, 7, 8, 9]))
    var_06_winter = var_06.sel(time=var_06.time.dt.month.isin([12, 1, 2, 3]))

    var_05_summer = var_05.sel(time=var_05.time.dt.month.isin([6, 7, 8, 9]))
    var_05_winter = var_05.sel(time=var_05.time.dt.month.isin([12, 1, 2, 3]))
    
    var_obs_imerg_summer = var_obs_imerg.sel(time=var_obs_imerg.time.dt.month.isin([6, 7, 8, 9]))
    var_obs_imerg_winter = var_obs_imerg.sel(time=var_obs_imerg.time.dt.month.isin([12, 1, 2, 3]))

    var_obs_cpc_summer = var_obs_cpc.sel(time=var_obs_cpc.time.dt.month.isin([6, 7, 8, 9]))
    var_obs_cpc_winter = var_obs_cpc.sel(time=var_obs_cpc.time.dt.month.isin([12, 1, 2, 3]))

    var_obs_era5_summer = var_obs_era5.sel(time=var_obs_era5.time.dt.month.isin([6, 7, 8, 9]))
    var_obs_era5_winter = var_obs_era5.sel(time=var_obs_era5.time.dt.month.isin([12, 1, 2, 3]))

    var_obs_mswep_summer = var_obs_mswep.sel(time=var_obs_mswep.time.dt.month.isin([6, 7, 8, 9]))
    var_obs_mswep_winter = var_obs_mswep.sel(time=var_obs_mswep.time.dt.month.isin([12, 1, 2, 3]))


# Summer Season

    evolve_var_08_su  = var_08_summer.mean("time")
    evolve_var_08_win = var_08_winter.mean("time")
    
    evolve_var_06_su  = var_06_summer.mean("time")
    evolve_var_06_win = var_06_winter.mean("time") 

    evolve_var_05_su  = var_05_summer.mean("time")
    evolve_var_05_win = var_05_winter.mean("time")

    evolve_var_obs_imerg_su  = var_obs_imerg_summer.mean("time") 
    evolve_var_obs_imerg_win = var_obs_imerg_winter.mean("time") 

    evolve_var_obs_cpc_su  = var_obs_cpc_summer.mean("time")
    evolve_var_obs_cpc_win = var_obs_cpc_winter.mean("time")

    evolve_var_obs_era5_su  = var_obs_era5_summer.mean("time") * 86400
    evolve_var_obs_era5_win = var_obs_era5_winter.mean("time") * 86400

    evolve_var_obs_mswep_su  = var_obs_mswep_summer.mean("time") 
    evolve_var_obs_mswep_win = var_obs_mswep_winter.mean("time") 


#  The Differnce between Local Summer and Local Winter
    diff_sum_win_08     = abs((evolve_var_08_su - evolve_var_08_win) * mask_08)
    diff_sum_win_06     = abs((evolve_var_06_su - evolve_var_06_win) * mask_06)
    diff_sum_win_05     =  abs((evolve_var_05_su - evolve_var_05_win) * mask_05)
    diff_sum_win_obs_imerg    = abs((evolve_var_obs_imerg_su - evolve_var_obs_imerg_win) * mask_imerg)
    diff_sum_win_obs_cpc      = abs((evolve_var_obs_cpc_su - evolve_var_obs_cpc_win) * mask_cpc)
    diff_sum_win_obs_era5     = abs((evolve_var_obs_era5_su - evolve_var_obs_era5_win) * mask_era5)
    diff_sum_win_obs_mswep     = abs((evolve_var_obs_mswep_su - evolve_var_obs_mswep_win) * mask_mswep)

    print(diff_sum_win_08.shape)
 
    print(diff_sum_win_obs_mswep.max()) 

    diff_sum_win_obs_mswep_nan = diff_sum_win_obs_mswep.where(diff_sum_win_obs_mswep != 0)
    diff_sum_win_obs_imerg_nan = diff_sum_win_obs_imerg.where(diff_sum_win_obs_imerg != 0)

    # Smooth the data array with a Gaussian filter (sigma=3)
    smoothed_data_08 = ndimage.gaussian_filter(diff_sum_win_08, sigma=3)
    smoothed_data_06 = ndimage.gaussian_filter(diff_sum_win_06, sigma=3)
    smoothed_data_05 = ndimage.gaussian_filter(diff_sum_win_05, sigma=3)
    
    smoothed_data_imerg = ndimage.gaussian_filter(diff_sum_win_obs_imerg, sigma=3)
    smoothed_data_era5 = ndimage.gaussian_filter(diff_sum_win_obs_era5, sigma=3)
    smoothed_data_mswep = ndimage.gaussian_filter(diff_sum_win_obs_mswep, sigma=3)
    smoothed_data_cpc = ndimage.gaussian_filter(diff_sum_win_obs_cpc, sigma=3)


    # Create a new xarray DataArray with the smoothed data, preserving coordinates and attributes
    diff_sum_win_08_smoothed = xr.DataArray(
    smoothed_data_08,
    coords=diff_sum_win_08.coords,
    dims=diff_sum_win_08.dims,
    attrs=diff_sum_win_08.attrs
    )

    diff_sum_win_06_smoothed = xr.DataArray(
    smoothed_data_06,
    coords=diff_sum_win_06.coords,
    dims=diff_sum_win_06.dims,
    attrs=diff_sum_win_06.attrs
    )

    diff_sum_win_05_smoothed = xr.DataArray(
    smoothed_data_05,
    coords=diff_sum_win_05.coords,
    dims=diff_sum_win_05.dims,
    attrs=diff_sum_win_05.attrs
    )


    diff_sum_win_obs_era5_smoothed = xr.DataArray(
    smoothed_data_era5,
    coords=diff_sum_win_obs_era5.coords,
    dims=diff_sum_win_obs_era5.dims,
    attrs=diff_sum_win_obs_era5.attrs
    )


    diff_sum_win_obs_mswep_smoothed = xr.DataArray(
    smoothed_data_mswep,
    coords=diff_sum_win_obs_mswep.coords,
    dims=diff_sum_win_obs_mswep.dims,
    attrs=diff_sum_win_obs_mswep.attrs
    )


    diff_sum_win_obs_cpc_smoothed = xr.DataArray(
    smoothed_data_cpc,
    coords=diff_sum_win_obs_cpc.coords,
    dims=diff_sum_win_obs_cpc.dims,
    attrs=diff_sum_win_obs_cpc.attrs
    )


    diff_sum_win_obs_imerg_smoothed = xr.DataArray(
    smoothed_data_imerg,
    coords=diff_sum_win_obs_imerg.coords,
    dims=diff_sum_win_obs_imerg.dims,
    attrs=diff_sum_win_obs_imerg.attrs
    )



#     diff_sum_win_08 = scipy.ndimage.zoom(diff_sum_win_08, 3)
#     diff_sum_win_06 = scipy.ndimage.zoom(diff_sum_win_06, 3)
#     diff_sum_win_05 = scipy.ndimage.zoom(diff_sum_win_05, 3)
#     diff_sum_win_obs_mswep = scipy.ndimage.zoom(diff_sum_win_obs_mswep, 3)



#  Set less than 1 to NA
     
# Absolute Percentage Differences

#     diff_sum_win_08_obs  = abs(diff_sum_win_08) - abs(diff_sum_win_obs_imerg) 
#     diff_sum_win_06_obs  = abs(diff_sum_win_06) - abs(diff_sum_win_obs_imerg)
#     diff_sum_win_05_obs  = abs(diff_sum_win_05) - abs(diff_sum_win_obs_imerg) 
#     diff_sum_win_obs_imerg_era5  = abs(diff_sum_win_obs_era5) - abs(diff_sum_win_obs_imerg) 
#     diff_sum_win_obs_cpc_era5  = abs(diff_sum_win_obs_cpc) - abs(diff_sum_win_obs_imerg) 
#     diff_sum_win_obs_mswep_era5  = abs(diff_sum_win_obs_mswep) - abs(diff_sum_win_obs_imerg)


#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
    ax = fig.add_subplot(3,2,i, projection=ccrs.PlateCarree())
     # Plot countours for 10KM, 40KM and 80KM
    diff_sum_win_08_smoothed.plot.contour(ax=ax,levels=[2.0], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5) 
     
    diff_sum_win_06_smoothed.plot.contour(ax=ax,levels=[2.0], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)

    diff_sum_win_05_smoothed.plot.contour(ax=ax,levels=[2.0], transform=ccrs.PlateCarree(), colors=['b'],linewidths=2.5, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_05.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['b'],linewidths=2.5)


    if i == 4:
    # First plot without colorbar
         plot_obj = diff_sum_win_obs_imerg_nan.plot(ax=ax, transform=ccrs.PlateCarree(), 
                                              cmap='Spectral_r', vmin=0, vmax=16,
                                              add_colorbar=False, add_labels=False)
    # Then add colorbar manually with bold formatting
         cbar = plt.colorbar(plot_obj, ax=ax, extend='both', shrink=1.2, orientation='vertical')
         cbar.set_label('IMERG Precipitation (mm/day)', weight='bold', fontsize=12)


#    if i == 4:
#	     			diff_sum_win_obs_imerg_nan.plot(ax=ax,transform=ccrs.PlateCarree(), cmap='Spectral_r',vmin=0,vmax=16,cbar_kwargs={'label': 'Total Precipitation (mm/day)', 'extend': 'both',  'shrink': 1.8, 'orientation': 'vertical', 'labelweight': 'bold'},add_labels=False)

    else:
        diff_sum_win_obs_imerg_nan.plot(ax=ax,transform=ccrs.PlateCarree(), cmap='Spectral_r',vmin=0,vmax=16,add_colorbar=False,add_labels=False)


#    ax.contourf(
#    diff_sum_win_obs_mswep['lon'], 
#    diff_sum_win_obs_mswep['lat'], 
#    diff_sum_win_obs_mswep.values, 
#    levels=[2.45, 2.55], 
#    hatches=['xx'], 
#    colors='none',  # important: no fill color
#    transform=ccrs.PlateCarree()
#    )


    diff_sum_win_obs_imerg_smoothed.plot.contour(ax=ax,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=2.5, add_colorbar=False, add_labels=False)

#    h1,_ = plot1.legend_elements()
#    h2,_ = plot2.legend_elements()
#    h3,_ = plot3.legend_elements()
#    h4,_ = plot4.legend_elements()

#    ax.legend([h1[0], h2[0],h3[0],h4[0]],  ['ICON-10KM', 'ICON-40KM','ICON-80KM','MSWEP'], loc='lower right',prop={'size': 12})

#    ax.set_xlabel("")
#    ax.set_ylabel("")
   # ax.set_xticks([])
   # ax.set_yticks([])

# Add coast line 
    ax.coastlines(resolution='10m',color='grey',linewidth=1)
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

    if i == 6:
        legend_handles = [Line2D([0], [0], color=c, lw=2.5) for c in colors]
        #ax.legend(legend_handles, labels, loc='lower left', fontsize=14,fontweight='bold')
        ax.legend(legend_handles, labels, loc='lower left', 
              prop={'size': 12, 'weight': 'bold'})


#plt.tight_layout()
plt.savefig(plot_name)
fig.savefig("Fig2_Monsoon_domains.pdf", bbox_inches='tight', pad_inches=0.1)
