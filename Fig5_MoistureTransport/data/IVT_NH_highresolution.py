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

data_base_dir_08 = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/R2B8/Transport'
model_data_08    =  'flux_r2b8.nc'

data_base_dir_08_mag = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/R2B8/Transport'
model_data_08_mag    =  'mag_flux_r2b8.nc'

# directory of 40KM simulation
data_base_dir_06 = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/R2B6/Transport'
model_data_06    =  'flux_r2b6.nc'


data_base_dir_06_mag = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/R2B6/Transport'
model_data_06_mag    =  'mag_flux_r2b6.nc'


# Directory for 80KM simulation

data_base_dir_05 = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/R2B5/Transport'
model_data_05    =  'flux_r2b5.nc'


data_base_dir_05_mag = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/R2B5/Transport'
model_data_05_mag    =  'mag_flux_r2b5.nc'


# Observational Data Sets for ERA5

data_base_dir_obs_era5 = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/ERA5/Transport'
model_data_obs_era5    =  'flux_era5.nc'


data_base_dir_obs_era5_mag = '/capstor/scratch/cscs/ppothapa/IITM_DATAREQUEST/ERA5/Transport'
model_data_obs_era5_mag    =  'mag_flux_era5.nc'


# netCDF variable name of variable to plot
var='qv','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_obs_era5    = os.path.join(data_base_dir_obs_era5, model_data_obs_era5)

model_data_dir_08_mag    = os.path.join(data_base_dir_08_mag, model_data_08_mag)
model_data_dir_06_mag    = os.path.join(data_base_dir_06_mag, model_data_06_mag)
model_data_dir_05_mag    = os.path.join(data_base_dir_05_mag, model_data_05_mag)
model_data_dir_obs_era5_mag    = os.path.join(data_base_dir_obs_era5_mag, model_data_obs_era5_mag)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_era5  = xr.open_dataset(model_data_dir_obs_era5)


##
ds_08_mag  = xr.open_dataset(model_data_dir_08_mag)
ds_06_mag  = xr.open_dataset(model_data_dir_06_mag)
ds_05_mag  = xr.open_dataset(model_data_dir_05_mag)
ds_obs_era5_mag  = xr.open_dataset(model_data_dir_obs_era5_mag)

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

for x in reg:

    if x == 'South_Asia':
    # Select a Region
    #South Asia
        lat_min=-15
        lat_max=60
        lon_min=30
        lon_max=150
        plot_name='South_Asia'
        i=1

    elif x == 'North Africa':
        lat_min=-20
        lat_max=30
        lon_min=-40
        lon_max=40
        plot_name='North Africa'
        i=2

    elif x == 'North America':
        lat_min=0
        lat_max=30
        lon_min=-120
        lon_max=-75
        plot_name='North America'
        i=3
    
    plot_name = 'NH_' + 'Moisture_Transport'
    print(plot_name)

# Read the Variable from Here.  
    var_08_qu = ds_08['qu']
    var_06_qu = ds_06['qu']
    var_05_qu = ds_05['qu'] 
    var_obs_era5_qu = ds_obs_era5['qu']

    var_08_qv = ds_08['qvv']
    var_06_qv = ds_06['qvv']
    var_05_qv = ds_05['qvv']
    var_obs_era5_qv = ds_obs_era5['qv']

## 
    var_08_mag = ds_08_mag['flxmag']
    var_06_mag = ds_06_mag['flxmag']
    var_05_mag = ds_05_mag['flxmag']
    var_obs_era5_mag = ds_obs_era5_mag['flxmag']

    plev_pa = ds_obs_era5['plev'] * 100
    g = 9.80665  # m/s²


# Select Lat & Lon focussing on the tropics 
    var_08_qu = var_08_qu.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_qu = var_06_qu.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_qu = var_05_qu.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_era5_qu = var_obs_era5_qu.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

    var_08_qv = var_08_qv.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_qv = var_06_qv.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_qv = var_05_qv.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_era5_qv = var_obs_era5_qv.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


    var_08_mag = var_08_mag.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06_mag = var_06_mag.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05_mag = var_05_mag.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_obs_era5_mag = var_obs_era5_mag.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select the Seasons to Caluculate the Monsoon Onset & Do Pentad Centered over a day!      
    var_08_qu_summer = var_08_qu.sel(time=var_08_qu.time.dt.month.isin([6,7,8,9]))
    var_08_qu_summer = xr.apply_ufunc(np.trapz,var_08_qu_summer,var_08_qu_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True,kwargs=None) / g
    var_08_qu_summer_clim = var_08_qu_summer.mean(dim="time")
    var_08_qu_summer_clim.name = "qu_08"
    filename = f"{x}_var_08_qu_summer_clim.nc"
    var_08_qu_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
#    var_08_qu_summer_clim.to_netcdf("var_08_qu_summer_clim.nc")


    var_06_qu_summer = var_06_qu.sel(time=var_06_qu.time.dt.month.isin([6,7,8,9]))
    var_06_qu_summer = xr.apply_ufunc(np.trapz,var_06_qu_summer,var_06_qu_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_06_qu_summer_clim = var_06_qu_summer.mean(dim="time")
    var_06_qu_summer_clim.name = "qu_06"
    filename = f"{x}_var_06_qu_summer_clim.nc"
    var_06_qu_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
#    var_06_qu_summer_clim.to_netcdf("var_06_qu_summer_clim.nc")



    var_05_qu_summer = var_05_qu.sel(time=var_05_qu.time.dt.month.isin([6,7,8,9]))
    var_05_qu_summer = xr.apply_ufunc(np.trapz,var_05_qu_summer,var_05_qu_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None)   / g
    var_05_qu_summer_clim = var_05_qu_summer.mean(dim="time") 
    var_05_qu_summer_clim.name = "qu_05"
    filename = f"{x}_var_05_qu_summer_clim.nc"
    var_05_qu_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
#    var_05_qu_summer_clim.to_netcdf("var_05_qu_summer_clim.nc")


    var_obs_era5_qu_summer = var_obs_era5_qu.sel(time=var_obs_era5_qu.time.dt.month.isin([6,7,8,9]))
    var_obs_era5_qu_summer = xr.apply_ufunc(np.trapz,var_obs_era5_qu_summer,plev_pa,
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_obs_era5_qu_summer_clim = var_obs_era5_qu_summer.mean(dim="time")
    var_obs_era5_qu_summer_clim.name = "qu_era5"
    filename = f"{x}_var_obs_era5_qu_summer_clim.nc"
    var_obs_era5_qu_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
# 

    var_08_qv_summer = var_08_qv.sel(time=var_08_qv.time.dt.month.isin([6,7,8,9]))
    var_08_qv_summer = xr.apply_ufunc(np.trapz,var_08_qv_summer,var_08_qv_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_08_qv_summer_clim = var_08_qv_summer.mean(dim="time")
    var_08_qv_summer_clim.name = "qv_08"
    filename = f"{x}_var_08_qv_summer_clim.nc"
    var_08_qv_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")    

#    var_08_qv_summer_clim.to_netcdf("var_08_qv_summer_clim.nc")


    var_06_qv_summer = var_06_qv.sel(time=var_06_qv.time.dt.month.isin([6,7,8,9]))
    var_06_qv_summer = xr.apply_ufunc(np.trapz,var_06_qv_summer,var_06_qv_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_06_qv_summer_clim = var_06_qv_summer.mean(dim="time")
    var_06_qv_summer_clim.name = "qv_06"
    filename = f"{x}_var_06_qv_summer_clim.nc"
    var_06_qv_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")



    var_05_qv_summer = var_05_qv.sel(time=var_05_qv.time.dt.month.isin([6,7,8,9]))
    var_05_qv_summer = xr.apply_ufunc(np.trapz,var_05_qv_summer,var_05_qv_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_05_qv_summer_clim = var_05_qv_summer.mean(dim="time")
    var_05_qv_summer_clim.name = "qv_05"
    filename = f"{x}_var_05_qv_summer_clim.nc"
    var_05_qv_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
#    var_05_qv_summer_clim.to_netcdf("var_05_qv_summer_clim.nc")

    var_obs_era5_qv_summer = var_obs_era5_qv.sel(time=var_obs_era5_qv.time.dt.month.isin([6,7,8,9]))
    var_obs_era5_qv_summer = xr.apply_ufunc(np.trapz,var_obs_era5_qv_summer,plev_pa,
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_obs_era5_qv_summer_clim = var_obs_era5_qv_summer.mean(dim="time")
    var_obs_era5_qv_summer_clim.name = "qv_era5"
    filename = f"{x}_var_obs_era5_qv_summer_clim.nc"
    var_obs_era5_qv_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")

    var_08_mag_summer = var_08_mag.sel(time=var_08_mag.time.dt.month.isin([6,7,8,9]))
    var_08_mag_summer = xr.apply_ufunc(np.trapz,var_08_mag_summer,var_08_mag_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_08_mag_summer_clim = var_08_mag_summer.mean(dim="time")
    var_08_mag_summer_clim.name = "mag_08"
    filename = f"{x}_var_08_mag_summer_clim.nc"
    var_08_mag_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
#    var_08_mag_summer_clim.to_netcdf("var_08_mag_summer_clim.nc")


    var_06_mag_summer = var_06_mag.sel(time=var_06_mag.time.dt.month.isin([6,7,8,9]))
    var_06_mag_summer = xr.apply_ufunc(np.trapz,var_06_mag_summer,var_06_mag_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_06_mag_summer_clim = var_06_mag_summer.mean(dim="time")
    var_06_mag_summer_clim.name = "mag_06"
    filename = f"{x}_var_06_mag_summer_clim.nc"
    var_06_mag_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")



    var_05_mag_summer = var_05_mag.sel(time=var_05_mag.time.dt.month.isin([6,7,8,9]))
    var_05_mag_summer = xr.apply_ufunc(np.trapz,var_05_mag_summer,var_05_mag_summer['plev'],
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_05_mag_summer_clim = var_05_mag_summer.mean(dim="time")
    var_05_mag_summer_clim.name = "mag_05"
    filename = f"{x}_var_05_mag_summer_clim.nc"
    var_05_mag_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")
  

    var_obs_era5_mag_summer = var_obs_era5_mag.sel(time=var_obs_era5_mag.time.dt.month.isin([6,7,8,9]))
    var_obs_era5_mag_summer = xr.apply_ufunc(np.trapz,var_obs_era5_mag_summer,plev_pa,
    input_core_dims=[['plev'], ['plev']], vectorize=True, kwargs=None) / g
    var_obs_era5_mag_summer_clim = var_obs_era5_mag_summer.mean(dim="time") 
    var_obs_era5_mag_summer_clim.name = "mag_obs_era5"
    filename = f"{x}_var_obs_era5_mag_summer_clim.nc"
    var_obs_era5_mag_summer_clim.to_netcdf(filename)
    print(f"Saved {filename}")

    cmap = mpl.colormaps['Spectral_r']

#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
    
    if i == 1:
             i = 1
             arrow_scale=2
             skip_value=10
             bounds = np.linspace(0,0.025,20)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    elif i == 2: 
             i = 5
             skip_value=5
             arrow_scale=0.6
             bounds = np.linspace(0,0.0025,20)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    elif i == 3: 
             i = 9
             skip_value=5
             arrow_scale=0.5
             bounds = np.linspace(0,0.01,20)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)


    ax1 = fig.add_subplot(3,4,i, projection=ccrs.PlateCarree())
     # Plot countours for 10KM, 40KM and 80KM
    var_08_mag_summer_clim.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
    # 3. Add quiver — optionally slice to avoid overcrowding
    skip = skip_value  # adjust to thin arrows
    ax1.quiver(
    var_08_qu_summer['lon'][::skip], 
    var_08_qu_summer['lat'][::skip],
    var_08_qu_summer_clim.values[::skip, ::skip], 
    var_08_qv_summer_clim.values[::skip, ::skip],
    transform=ccrs.PlateCarree(), 
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )


#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5) 
    
    ax2 = fig.add_subplot(3,4,i+1, projection=ccrs.PlateCarree()) 
    var_06_mag_summer_clim.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)
    skip = skip_value  # adjust to thin arrows
    ax2.quiver(
    var_06_qu_summer['lon'][::skip],
    var_06_qu_summer['lat'][::skip],
    var_06_qu_summer_clim.values[::skip, ::skip],
    var_06_qv_summer_clim.values[::skip, ::skip],
    transform=ccrs.PlateCarree(),
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )   




    ax3 = fig.add_subplot(3,4,i+2, projection=ccrs.PlateCarree())
    var_05_mag_summer_clim.plot(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_05.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['b'],linewidths=2.5)
    skip = skip_value  # adjust to thin arrows
    ax3.quiver(
    var_05_qu_summer['lon'][::skip],
    var_05_qu_summer['lat'][::skip],
    var_05_qu_summer_clim.values[::skip, ::skip],
    var_05_qv_summer_clim.values[::skip, ::skip],
    transform=ccrs.PlateCarree(),
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )



    ax4 = fig.add_subplot(3,4,i+3, projection=ccrs.PlateCarree())
    var_obs_era5_mag_summer_clim.plot(ax=ax4, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, cbar_kwargs={'pad': 0.02})
    skip = skip_value  # adjust to thin arrows
    ax4.quiver(
    var_obs_era5_qu_summer['lon'][::skip],
    var_obs_era5_qu_summer['lat'][::skip],
    var_obs_era5_qu_summer_clim.values[::skip, ::skip],
    var_obs_era5_qv_summer_clim.values[::skip, ::skip],
    transform=ccrs.PlateCarree(),
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )



# Add coast line 

# Title and axis labels
    

    ax4.coastlines(resolution='10m',color='black',linewidth=2)
    ax4.set_title('ERA5',fontweight='bold', fontsize=10)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.xaxis.set_tick_params(labelsize=2)
    ax4.yaxis.set_tick_params(labelsize=2)
    ax4.set_aspect('auto')
    gls = ax4.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False

    ax1.coastlines(resolution='10m',color='black',linewidth=2)
#    ax1.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    
    ax1.set_title('ICON (10KM)',fontweight='bold', fontsize=10)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.xaxis.set_tick_params(labelsize=2)
    ax1.yaxis.set_tick_params(labelsize=2)
    ax1.set_aspect('auto')
    gls = ax1.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False


    ax2.coastlines(resolution='10m',color='black',linewidth=2)
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

    ax3.coastlines(resolution='10m',color='black',linewidth=2)
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

#plt.tight_layout()
plt.savefig(plot_name)
