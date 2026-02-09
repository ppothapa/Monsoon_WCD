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


from scipy.signal import butter, filtfilt

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
model_data_08    =  'u_all_level_rmp_50_day_JJAS_cli.nc'
model_data_08_v  =  'v_all_level_rmp_50_day_JJAS.nc'


# directory of 40KM simulation
data_base_dir_06 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/Transport'
model_data_06    =  'u_all_level_rmp_50_day_JJAS_cli.nc'
model_data_06_v  =  'v_all_level_rmp_50_day_JJAS.nc'

# Directory for 80KM simulation

data_base_dir_05 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/Transport'
model_data_05    =  'u_all_level_rmp_50_day_JJAS_cli.nc'
model_data_05_v  =  'v_all_level_rmp_50_day_JJAS.nc'

# Observational Data Sets for ERA5

data_base_dir_obs_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/Transport'
model_data_obs_era5      =  'u_era5_all_levels_cli.nc'
model_data_obs_era5_v    =  'v_era5_all_levels_day_JJAS.nc'
## Here you load the V wind!

# netCDF variable name of variable to plot
var='qv','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_obs_era5    = os.path.join(data_base_dir_obs_era5, model_data_obs_era5)


##  For the V wind

model_data_dir_08_v    = os.path.join(data_base_dir_08, model_data_08_v)
model_data_dir_06_v    = os.path.join(data_base_dir_06, model_data_06_v)
model_data_dir_05_v    = os.path.join(data_base_dir_05, model_data_05_v)
model_data_dir_obs_era5_v    = os.path.join(data_base_dir_obs_era5, model_data_obs_era5_v)


#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_obs_era5  = xr.open_dataset(model_data_dir_obs_era5)

## 


ds_08_v  = xr.open_dataset(model_data_dir_08_v)
ds_06_v  = xr.open_dataset(model_data_dir_06_v)
ds_05_v  = xr.open_dataset(model_data_dir_05_v)
ds_obs_era5_v  = xr.open_dataset(model_data_dir_obs_era5_v)


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

#for Lat/lonfor the Meridional Variance

lat_min_v=-5
lat_max_v=35
lon_min_v=-20
lon_max_v=50
plot_name_v='North_Africa_variance'


# Read the Variable from Here.  
var_08_u = ds_08['u']
var_06_u = ds_06['u']
var_05_u = ds_05['u'] 
var_obs_era5_u = ds_obs_era5['u']

### 

var_08_v = ds_08_v['v']
var_06_v = ds_06_v['v']
var_05_v = ds_05_v['v']
var_obs_era5_v = ds_obs_era5_v['v']

###

# Select Lat & Lon focussing on the tropics 
var_08_v = var_08_v.sel(lat=slice(lat_min_v,lat_max_v),lon=slice(lon_min_v,lon_max_v))
var_06_v = var_06_v.sel(lat=slice(lat_min_v,lat_max_v),lon=slice(lon_min_v,lon_max_v))
var_05_v = var_05_v.sel(lat=slice(lat_min_v,lat_max_v),lon=slice(lon_min_v,lon_max_v))
var_obs_era5_v = var_obs_era5_v.sel(lat=slice(lat_min_v,lat_max_v),lon=slice(lon_min_v,lon_max_v))

##  

var_08_u = var_08_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
var_06_u = var_06_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
var_05_u = var_05_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
var_obs_era5_u = var_obs_era5_u.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))


# Select the Seasons to Caluculate the Monsoon Onset & Do Pentad Centered over a day!      
#var_08_u_summer = var_08_u.sel(time=var_08_u.time.dt.month.isin([6,7,8,9]))
var_08_u_summer_clim = var_08_u.mean(dim=["lon","time"])
var_08_u_summer_clim = var_08_u_summer_clim.sortby('plev', ascending=False)

###  

#var_08_v_summer = var_08_v.sel(time=var_08_v.time.dt.month.isin([6,7,8,9]),plev=70000)
var_08_v_summer = var_08_v.sel(plev=70000)
var_08_v_summer_intra = apply_bandpass(var_08_v_summer) 
var_08_v_summer_clim = var_08_v_summer_intra.var(dim='time', skipna=True)     


###  
#var_06_u_summer = var_06_u.sel(time=var_06_u.time.dt.month.isin([6,7,8,9]))
var_06_u_summer_clim = var_06_u.mean(dim=["lon","time"])
var_06_u_summer_clim = var_06_u_summer_clim.sortby('plev', ascending=False)

#var_06_v_summer = var_06_v.sel(time=var_06_v.time.dt.month.isin([6,7,8,9]),plev=70000)
var_06_v_summer = var_06_v.sel(plev=70000)
var_06_v_summer_intra = apply_bandpass(var_06_v_summer)
var_06_v_summer_clim = var_06_v_summer_intra.var(dim='time', skipna=True)

##
#var_05_u_summer = var_05_u.sel(time=var_05_u.time.dt.month.isin([6,7,8,9]))
var_05_u_summer_clim = var_05_u.mean(dim=["lon","time"]) 
var_05_u_summer_clim = var_05_u_summer_clim.sortby('plev', ascending=False)

#
#var_05_v_summer = var_05_v.sel(time=var_05_v.time.dt.month.isin([6,7,8,9]),plev=70000)
var_05_v_summer = var_05_v.sel(plev=70000)
var_05_v_summer_intra = apply_bandpass(var_05_v_summer)
var_05_v_summer_clim = var_05_v_summer_intra.var(dim='time', skipna=True)



##
#var_obs_era5_u_summer = var_obs_era5_u.sel(time=var_obs_era5_u.time.dt.month.isin([6,7,8,9]))
var_obs_era5_u_summer_clim = var_obs_era5_u.mean(dim=["lon","time"])
var_obs_era5_u_summer_clim = var_obs_era5_u_summer_clim.sortby('plev', ascending=False)

####  

#var_obs_era5_v_summer = var_obs_era5_v.sel(time=var_obs_era5_v.time.dt.month.isin([6,7,8,9]),plev=700)
var_obs_era5_v_summer = var_obs_era5_v.sel(plev=700)
var_obs_era5_v_summer_intra = apply_bandpass(var_obs_era5_v_summer)
var_obs_era5_v_summer_clim = var_obs_era5_v_summer_intra.var(dim='time', skipna=True)


# Bias calculation


var_08_v_summer_clim_bias = var_08_v_summer_clim - var_obs_era5_v_summer_clim
var_06_v_summer_clim_bias = var_06_v_summer_clim - var_obs_era5_v_summer_clim
var_05_v_summer_clim_bias = var_05_v_summer_clim - var_obs_era5_v_summer_clim



# 
#    cmap = mpl.colormaps['Spectral_r']
cmap = mpl.colormaps['bwr']
cmap_intra = mpl.colormaps['Spectral_r']
cmap_intra_bias = mpl.colormaps['bwr']

bounds = np.arange(-20,20.1,2)
bounds_intra = np.arange(1,15.1,1)
bounds_intra_bias = np.arange(-6,6.1,1)


norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)
norm_intra = mcolors.BoundaryNorm(boundaries=bounds_intra, ncolors=256)
norm_intra_bias = mcolors.BoundaryNorm(boundaries=bounds_intra_bias, ncolors=256)


proj = ccrs.Mercator()
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


ax2 = fig.add_subplot(3,4,2)
# Plot contours for 10KM, 40KM and 80KM
cf2 = var_08_u_summer_clim.plot.contourf(ax=ax2, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
cs2 = var_08_u_summer_clim.plot.contour(ax=ax2,levels=[-8, -7, -6],colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
ax2.clabel(cs2, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
ax2.axvline(x=5, color='red', linewidth=2.5, linestyle='--')
ax2.axvline(x=9.5, color='red', linewidth=2.5, linestyle='--')
ax2.axvline(x=10, color='black', linewidth=2.5, linestyle='--')
ax2.axvline(x=18, color='black', linewidth=2.5, linestyle='--')

ax3 = fig.add_subplot(3,4,3)
cf3 = var_06_u_summer_clim.plot.contourf(ax=ax3, cmap=cmap, x="lat",y="plev", norm=norm, add_colorbar=False, add_labels=False,levels=[-8, -7, -6])
cs3 = var_06_u_summer_clim.plot.contour(ax=ax3,levels=[-8, -7, -6],colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
ax3.clabel(cs3, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
ax3.axvline(x=5, color='red', linewidth=2.5, linestyle='--')
ax3.axvline(x=9.5, color='red', linewidth=2.5, linestyle='--')
ax3.axvline(x=10, color='black', linewidth=2.5, linestyle='--')
ax3.axvline(x=18, color='black', linewidth=2.5, linestyle='--')

ax4 = fig.add_subplot(3,4,4)
cf4 = var_05_u_summer_clim.plot.contourf(ax=ax4, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False ,levels=[-8, -7, -6])
cs4 = var_05_u_summer_clim.plot.contour(ax=ax4,levels=[-8, -7, -6],colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
ax4.clabel(cs4, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
ax4.axvline(x=5, color='red', linewidth=2.5, linestyle='--')
ax4.axvline(x=9.5, color='red', linewidth=2.5, linestyle='--')
ax4.axvline(x=10, color='black', linewidth=2.5, linestyle='--')
ax4.axvline(x=18, color='black', linewidth=2.5, linestyle='--')

cbar = plt.colorbar(cf4, ax=ax4, pad=0.02)
cbar.set_label('U-wind (m/s)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)
for label in cbar.ax.get_yticklabels():  # Use 'yticklabels' for vertical colorbar
    label.set_fontweight('bold')




ax1 = fig.add_subplot(3,4,1)
cf1 = var_obs_era5_u_summer_clim.plot.contourf(ax=ax1, cmap=cmap, x="lat",y="plev",norm=norm, add_colorbar=False, add_labels=False, levels=[-8, -7, -6])
cs1 = var_obs_era5_u_summer_clim.plot.contour(ax=ax1,levels=[-8, -7, -6],colors="black",linewidths=0.8,add_labels=False)
# Add contour labels
ax1.clabel(cs1, inline=True, fontsize=8, colors='black', fmt='%d')
# Add vertical lines
ax1.axvline(x=5, color='red', linewidth=2.5, linestyle='--')
ax1.axvline(x=9.5, color='red', linewidth=2.5, linestyle='--')
ax1.axvline(x=10, color='black', linewidth=2.5, linestyle='--')
ax1.axvline(x=18, color='black', linewidth=2.5, linestyle='--')

# Add coast line 

# Title and axis labels
    

#    ax4.coastlines(resolution='10m',color='black',linewidth=2)
ax1.set_title('(a) ERA5 (Reference)',fontweight='bold', fontsize=14)
ax1.set_xlabel('Latitude', fontsize=10)
ax1.set_ylabel('Pressure Level', fontweight='bold', fontsize=14)
ax1.xaxis.set_tick_params(labelsize=2)
ax1.yaxis.set_tick_params(labelsize=2)
ax1.set_aspect('auto')
ax1.xaxis.set_tick_params(labelsize=10)
ax1.yaxis.set_tick_params(labelsize=10)
ax1.invert_yaxis()
 
    
ax2.set_title('(b) ICON (10KM)',fontweight='bold', fontsize=14)
ax2.set_xlabel('Latitude')
ax2.set_ylabel('Pressure Level')
ax2.xaxis.set_tick_params(labelsize=2)
ax2.yaxis.set_tick_params(labelsize=2)
ax2.set_aspect('auto')
ax2.xaxis.set_tick_params(labelsize=10)
ax2.yaxis.set_tick_params(labelsize=10)
ax2.invert_yaxis()
ax2.set_ylabel('')
ax2.set_yticklabels([])



ax3.set_title('(c) ICON (40KM)',fontweight='bold', fontsize=14)
ax3.set_xlabel('Latitude')
ax3.set_ylabel('Pressure Level')
ax3.xaxis.set_tick_params(labelsize=2)
ax3.yaxis.set_tick_params(labelsize=2)
ax3.set_aspect('auto')
ax3.xaxis.set_tick_params(labelsize=10)
ax3.yaxis.set_tick_params(labelsize=10)
ax3.invert_yaxis()
ax3.set_ylabel('')
ax3.set_yticklabels([])



#    ax3.coastlines(resolution='10m',color='black',linewidth=2)
#    ax3.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
ax4.set_title('(d) ICON (80KM)',fontweight='bold', fontsize=14)
ax4.set_xlabel('Latitude')
ax4.set_ylabel('Pressure Level')
ax4.xaxis.set_tick_params(labelsize=2)
ax4.yaxis.set_tick_params(labelsize=2)
ax4.set_aspect('auto')
ax4.xaxis.set_tick_params(labelsize=10)
ax4.yaxis.set_tick_params(labelsize=10)
ax4.invert_yaxis()
ax4.set_ylabel('')
ax4.set_yticklabels([])

## Plot the second Row with v climatology:  axis of the 

ax5 = fig.add_subplot(3,4,5, projection=proj)
var_obs_era5_v_summer_clim.plot.contourf(ax=ax5, transform=ccrs.PlateCarree(), cmap=cmap_intra, norm=norm_intra, add_colorbar=False, add_labels=False)

# Draw a box from 6°N to 20°N and -14°E to 15°E
lon_min_var, lon_max_var = -14, 15
lat_min_var, lat_max_var = 6, 20

# Create the box coordinates
box_lons = [lon_min_var, lon_max_var, lon_max_var, lon_min_var, lon_min_var]
box_lats = [lat_min_var, lat_min_var, lat_max_var, lat_max_var, lat_min_var]


## Here Subtract the Box for flower plots. 

# Select Lat & Lon focussing on the tropics
var_08_v_summer_clim_bias_flower = var_08_v_summer_clim_bias.sel(lat=slice(lat_min_var,lat_max_var),lon=slice(lon_min_var,lon_max_var))
var_06_v_summer_clim_bias_flower = var_06_v_summer_clim_bias.sel(lat=slice(lat_min_var,lat_max_var),lon=slice(lon_min_var,lon_max_var))
var_05_v_summer_clim_bias_flower = var_05_v_summer_clim_bias.sel(lat=slice(lat_min_var,lat_max_var),lon=slice(lon_min_var,lon_max_var))


 # Prepare data for box plot
diff_data = [
  var_08_v_summer_clim_bias_flower.values.flatten(),
  var_06_v_summer_clim_bias_flower.values.flatten(),
  var_05_v_summer_clim_bias_flower.values.flatten()
   ]

  #
diff_data_clean = [d[~np.isnan(d)] for d in diff_data]





# Plot the box
ax5.plot(box_lons, box_lats, transform=ccrs.PlateCarree(), 
         color='black', linewidth=2, linestyle='-')



ax6 = fig.add_subplot(3,4,6, projection=proj)
     # Plot countours for 10KM, 40KM and 80KM
var_08_v_summer_clim.plot.contourf(ax=ax6, transform=ccrs.PlateCarree(), cmap=cmap_intra, norm=norm_intra, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5)

ax6.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')

ax7 = fig.add_subplot(3,4,7, projection=proj)
var_06_v_summer_clim.plot.contourf(ax=ax7, transform=ccrs.PlateCarree(), cmap=cmap_intra, norm=norm_intra, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)

ax7.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')

ax8 = fig.add_subplot(3,4,8, projection=proj)
cvar1 = var_05_v_summer_clim.plot.contourf(ax=ax8, transform=ccrs.PlateCarree(), cmap=cmap_intra, norm=norm_intra, add_colorbar=False, add_labels=False )
ax8.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')

cbar = plt.colorbar(cvar1, ax=ax8, pad=0.02)
cbar.set_label('Mean AEW activity (m2/s2)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)
# Make tick labels bold (consistent with previous)
for label in cbar.ax.get_yticklabels():  # Use 'yticklabels' for vertical colorbar
    label.set_fontweight('bold')



ax5.coastlines(resolution='10m',color='black',linewidth=1)
ax5.set_title('(e) ERA5 (Reference)',fontweight='bold', fontsize=14)
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
ax5.xaxis.set_tick_params(labelsize=2)
ax5.yaxis.set_tick_params(labelsize=2)
ax5.set_aspect('auto')
gls = ax5.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False


ax6.coastlines(resolution='10m',color='black',linewidth=1)
ax6.set_title('(f) ICON (10KM)',fontweight='bold', fontsize=14)
ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
ax6.xaxis.set_tick_params(labelsize=2)
ax6.yaxis.set_tick_params(labelsize=2)
ax6.set_aspect('auto')
gls = ax6.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False
ax6.set_ylabel('')
ax6.set_yticklabels([])


ax7.coastlines(resolution='10m',color='black',linewidth=1)
#    ax2.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
ax7.set_title('(g) ICON (40KM)',fontweight='bold', fontsize=14)
ax7.set_xlabel('Longitude')
ax7.set_ylabel('Latitude')
ax7.xaxis.set_tick_params(labelsize=2)
ax7.yaxis.set_tick_params(labelsize=2)
ax7.set_aspect('auto')
gls = ax7.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False
ax7.set_ylabel('')
ax7.set_yticklabels([])


ax8.coastlines(resolution='10m',color='black',linewidth=1)
#    ax3.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
ax8.set_title('(h) ICON (80KM)',fontweight='bold', fontsize=14)
ax8.set_xlabel('Longitude')
ax8.set_ylabel('Latitude')
ax8.xaxis.set_tick_params(labelsize=2)
ax8.yaxis.set_tick_params(labelsize=2)
ax8.set_aspect('auto')
gls = ax8.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False
#    _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)
ax8.set_ylabel('')
ax8.set_yticklabels([])


ax9 = fig.add_subplot(3,4,9, projection=proj)
     # Plot countours for 10KM, 40KM and 80KM
var_08_v_summer_clim_bias.plot.contourf(ax=ax9, transform=ccrs.PlateCarree(), cmap=cmap_intra_bias, norm=norm_intra_bias, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5)

ax9.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')

ax10 = fig.add_subplot(3,4,10, projection=proj)
var_06_v_summer_clim_bias.plot.contourf(ax=ax10, transform=ccrs.PlateCarree(), cmap=cmap_intra_bias, norm=norm_intra_bias, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)

ax10.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')


ax11 = fig.add_subplot(3,4,11, projection=proj)
var_05_v_summer_clim_bias.plot.contourf(ax=ax11, transform=ccrs.PlateCarree(), cmap=cmap_intra_bias, norm=norm_intra_bias, add_colorbar=False, add_labels=False)
ax11.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')


# Create colorbar for all three subplots
all_axes = [ax9, ax10, ax11]
cbar = fig.colorbar(all_axes[0].collections[0], ax=all_axes,
                  pad=0.02, fraction=0.06, aspect=25, extend='both', orientation='horizontal')

# Set colorbar properties
cbar.set_label('Bias in Mean AEW activity (m2/s2)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(axis='x', rotation=45, labelsize=10)
[label.set_fontweight('bold') for label in cbar.ax.get_xticklabels()]


ax9.coastlines(resolution='10m',color='black',linewidth=1)
ax9.set_title('(i) ICON (10KM) - ERA5',fontweight='bold', fontsize=14)
ax9.set_xlabel('Longitude')
ax9.set_ylabel('Latitude')
ax9.xaxis.set_tick_params(labelsize=2)
ax9.yaxis.set_tick_params(labelsize=2)
ax9.set_aspect('auto')
gls = ax9.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False


ax10.coastlines(resolution='10m',color='black',linewidth=1)
#    ax2.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
ax10.set_title('(j) ICON (40KM) - ERA5',fontweight='bold', fontsize=14)
ax10.set_xlabel('Longitude')
ax10.set_ylabel('Latitude')
ax10.xaxis.set_tick_params(labelsize=2)
ax10.yaxis.set_tick_params(labelsize=2)
ax10.set_aspect('auto')
gls = ax10.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False
ax10.set_ylabel('')
ax10.set_yticklabels([])

ax11.coastlines(resolution='10m',color='black',linewidth=1)
#    ax3.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
ax11.set_title('(k) ICON (80KM) - ERA5',fontweight='bold', fontsize=14)
ax11.set_xlabel('Longitude')
ax11.set_ylabel('Latitude')
ax11.xaxis.set_tick_params(labelsize=2)
ax11.yaxis.set_tick_params(labelsize=2)
ax11.set_aspect('auto')
gls = ax11.gridlines(draw_labels=True,color="none")
gls.top_labels=False
gls.right_labels=False
#    _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)
ax11.set_ylabel('')
ax11.set_yticklabels([])

all_stats = []

ax12 = fig.add_subplot(3,4,12)  # No projection for box plot
 # Create box plot
COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
violin_parts = ax12.violinplot(diff_data_clean, showmeans=True, showmedians=True)

labels = ['10km', '40km', '80km']
print("\n" + "="*60)
print("VIOLIN PLOT STATISTICS - {region_label} onset Differences (Core Monsoon Region)")
print("="*60)


for j, (data, res_label) in enumerate(zip(diff_data_clean, ['10km', '40km', '80km']), 1):

    if len(data) > 0:  # Check if data exists
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            q25, q75 = np.percentile(data, [25, 75])

            # Store stats for later use
            stats = {
                'resolution': res_label,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'q25': q25,
                'q75': q75,
                'min': data.min(),
                'max': data.max(),
                'sample_size': len(data)
            }
            all_stats.append(stats)

            print(f"\n{res_label}:")
            print(f"  Mean: {mean_val:5.2f} days")
            print(f"  Median: {median_val:5.2f} days")
            print(f"  Standard Deviation: {std_val:5.2f} days")
            print(f"  IQR (Q25-Q75): {q25:5.2f} to {q75:5.2f} days")
            print(f"  Range: {data.min():5.2f} to {data.max():5.2f} days")
            print(f"  Sample Size: {len(data):d} grid points")
    else:
            print(f"\n{res_label}: No valid data points")


ax12.set_xticks([1, 2, 3])
ax12.set_xticklabels(['10km', '40km', '80km'], fontweight='bold', fontsize=12)
ax12.set_ylabel('AEW Biases (m2/s2)', fontweight='bold', fontsize=12)
ax12.set_title('(l) Distribution AEW biases', fontweight='bold', fontsize=14)
ax12.grid(True, alpha=0.3)
ax12.axhline(y=0, color='grey', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
ax12.tick_params(axis='y', which='major', labelsize=11)
for label in ax12.get_yticklabels():
    label.set_fontweight('bold')


# Apply your color scheme
for pc, color in zip(violin_parts['bodies'], COLORS):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
violin_parts['cmedians'].set_color('black')
violin_parts['cmedians'].set_linewidth(2)
violin_parts['cmeans'].set_color('brown')
violin_parts['cmeans'].set_linewidth(2)

#    ax4.set_ylim(np.nanmin(all_diffs) - 5, np.nanmax(all_diffs) + 5)
ax12.set_ylim(-3,15)


#plt.tight_layout()
plt.savefig(plot_name)
fig.savefig("African_Waves.pdf", bbox_inches='tight', pad_inches=0.1)
