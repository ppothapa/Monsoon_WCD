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


##

# Color Scales Range
a = [0,.01,.1,.25,.5,1,1.5,2,3,4,6,8,10,15,20,30]
a = [0,.005,.01,.15,.25,0.5,1,2,3,4,5,6,7,8,20,30]
a = [-30,-20,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,20,30]
bounds = np.linspace(0,100,10)
#bounds = np.arange(1,10.1,1)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)


# The plotting call from here 

#fig = plt.figure(figsize=(16, 12), constrained_layout=True)
fig = plt.figure(figsize=(16, 12),constrained_layout=True)
#fig = plt.figure(figsize=(16, 12))


colors = ['r', 'g', 'b', 'k']
labels = ['ICON-10KM', 'ICON-40KM', 'ICON-80KM', 'OBS (MSWEP)']
Titles = ['Asian Summer Monsoon', 'North African Monsoon', 'North American Monsoon']

reg = 'South_Asia','North_Africa','North_America'

all_stats = []

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
        core='SAsiaM Core'
        region_label='SAsiaM'

        lat_min_core=18
        lat_max_core=25
        lon_min_core=72
        lon_max_core=85

    elif x == 'North_Africa':
        lat_min=-20
        lat_max=30
        lon_min=-40
        lon_max=40
        plot_name='North Africa'
        i=2
        core='WAfriM Core'
        region_label='NAfriM'
   
        lat_min_core=10
        lat_max_core=20
        lon_min_core=-18
        lon_max_core=16


    elif x == 'North_America':
        lat_min=0
        lat_max=35
        lon_min=-120
        lon_max=-75
        plot_name='North America'
        i=3
        core='NAmerM Core'
        region_label='NAmerM'

        lat_min_core=20
        lat_max_core=30
        lon_min_core=-110
        lon_max_core=-100
 
    plot_name = 'NH_' + 'Moisture_Transport'
    print(plot_name)
   
    filename = f"data/{x}_var_08_qu_summer_clim.nc"
    ds_08_qu  = xr.open_dataset(filename)
    
    filename = f"data/{x}_var_06_qu_summer_clim.nc"
    ds_06_qu  = xr.open_dataset(filename)

    filename = f"data/{x}_var_05_qu_summer_clim.nc"
    ds_05_qu  = xr.open_dataset(filename)

    filename = f"data/{x}_var_obs_era5_qu_summer_clim.nc"
    ds_obs_era5_qu  = xr.open_dataset(filename)
  

# Read the Variable from Here.  
    var_08_qu = ds_08_qu['qu_08']
    var_06_qu = ds_06_qu['qu_06']
    var_05_qu = ds_05_qu['qu_05'] 
    var_obs_era5_qu = ds_obs_era5_qu['qu_era5']

    diff_var_08_qu = var_08_qu - var_obs_era5_qu
    diff_var_06_qu = var_06_qu - var_obs_era5_qu
    diff_var_05_qu = var_05_qu - var_obs_era5_qu



    filename = f"data/{x}_var_08_qv_summer_clim.nc"
    ds_08_qv  = xr.open_dataset(filename)

    filename = f"data/{x}_var_06_qv_summer_clim.nc"
    ds_06_qv  = xr.open_dataset(filename)

    filename = f"data/{x}_var_05_qv_summer_clim.nc"
    ds_05_qv  = xr.open_dataset(filename)

    filename = f"data/{x}_var_obs_era5_qv_summer_clim.nc"
    ds_obs_era5_qv  = xr.open_dataset(filename)


    var_08_qv = ds_08_qv['qv_08']
    var_06_qv = ds_06_qv['qv_06']
    var_05_qv = ds_05_qv['qv_05']
    var_obs_era5_qv = ds_obs_era5_qv['qv_era5']

    diff_var_08_qv = var_08_qv - var_obs_era5_qv
    diff_var_06_qv = var_06_qv - var_obs_era5_qv
    diff_var_05_qv = var_05_qv - var_obs_era5_qv

    filename = f"data/{x}_var_08_mag_summer_clim.nc"
    ds_08_mag  = xr.open_dataset(filename)

    filename = f"data/{x}_var_06_mag_summer_clim.nc"
    ds_06_mag  = xr.open_dataset(filename)

    filename = f"data/{x}_var_05_mag_summer_clim.nc"
    ds_05_mag  = xr.open_dataset(filename)

    filename = f"data/{x}_var_obs_era5_mag_summer_clim.nc"
    ds_obs_era5_mag  = xr.open_dataset(filename)


## 
    var_08_mag = ds_08_mag['mag_08']
    var_06_mag = ds_06_mag['mag_06']
    var_05_mag = ds_05_mag['mag_05']
    var_obs_era5_mag = ds_obs_era5_mag['mag_obs_era5']

##  
    diff_var_08_mag = var_08_mag - var_obs_era5_mag
    diff_var_06_mag = var_06_mag - var_obs_era5_mag
    diff_var_05_mag = var_05_mag - var_obs_era5_mag

    diff_var_08_mag_core = diff_var_08_mag.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))

    diff_var_06_mag_core = diff_var_06_mag.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))

    diff_var_05_mag_core = diff_var_05_mag.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))

    ######

    # Create the box coordinates
    box_lons = [lon_min_core, lon_max_core, lon_max_core, lon_min_core, lon_min_core]
    box_lats = [lat_min_core, lat_min_core, lat_max_core, lat_max_core, lat_min_core]


#    original_cmap = mpl.colormaps['Spectral_r']
   
    # Create a new colormap where the bottom 10% is white, then transitions to Spectral_r
#    colors = []
#    for i in range(256):
#             if i < 25:  # First ~10% is white
#              colors.append('white')
#    else:
#        # Adjust the index to map the remaining colors to the original colormap
#        colors.append(original_cmap((i-25)/231))
#
#    new_cmap = mcolors.LinearSegmentedColormap.from_list('Spectral_r_white', colors, N=256)
#    cmap = new_cmap

    cmap = mpl.colormaps['bwr_r']
#    cmap.set_under('white')


 
#  Preparing for the figures & Spceifying the Projections
    proj = ccrs.Mercator()
    
    if i == 1:
             i = 1
             arrow_scale=1000
             skip_value=10
             bounds = np.linspace(-2,2,20)
             bounds = np.arange(-2,2.1,0.2)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)


    elif i == 2: 
             i = 5
             skip_value=8
             arrow_scale=700
             bounds = np.linspace(-0.5,0.5,20)
             bounds = np.arange(-0.5,0.51,0.05)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    elif i == 3: 
             i = 9
             skip_value=4
             arrow_scale=1000
             bounds = np.linspace(-1,1,20)
             bounds = np.arange(-1,1.1,0.1)
             norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    ax1 = fig.add_subplot(3,4,i, projection=ccrs.PlateCarree())
     # Plot countours for 10KM, 40KM and 80KM
    diff_var_08_mag.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
    # 3. Add quiver — optionally slice to avoid overcrowding

    ax1.plot(box_lons, box_lats, transform=ccrs.PlateCarree(),
         color='black', linewidth=2, linestyle='-')
    skip = skip_value  # adjust to thin arrows
    ax1.quiver(
    var_08_qu['lon'][::skip], 
    var_08_qu['lat'][::skip],
    diff_var_08_qu.values[::skip, ::skip], 
    diff_var_08_qv.values[::skip, ::skip],
    transform=ccrs.PlateCarree(), 
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )


#     plot = diff_sum_win_08.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['r'],linewidths=2.5) 
    
    ax2 = fig.add_subplot(3,4,i+1, projection=ccrs.PlateCarree()) 
    diff_var_06_mag.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, add_labels=False)
#     plot = diff_sum_win_06.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['g'],linewidths=2.5)
    skip = skip_value  # adjust to thin arrows
    ax2.quiver(
    var_06_qu['lon'][::skip],
    var_06_qu['lat'][::skip],
    diff_var_06_qu.values[::skip, ::skip],
    diff_var_06_qv.values[::skip, ::skip],
    transform=ccrs.PlateCarree(),
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )   



    ax3 = fig.add_subplot(3,4,i+2, projection=ccrs.PlateCarree())
    plot_obj = diff_var_05_mag.plot(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, cbar_kwargs={'pad': 0.02, 'label': 'Integrated Vertical Transport (Kg m-1 s-1)'})
    cbar = plot_obj.colorbar
    cbar.set_label('IVT (Kg m⁻¹ s⁻¹)', weight='bold', size=14)
#     plot = diff_sum_win_05.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['b'],linewidths=2.5)
#    diff_var_05_mag.plot(ax=ax3, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,add_colorbar=False, add_labels=False)
    skip = skip_value  # adjust to thin arrows
    ax3.quiver(
    var_05_qu['lon'][::skip],
    var_05_qu['lat'][::skip],
    diff_var_05_qu.values[::skip, ::skip],
    diff_var_05_qv.values[::skip, ::skip],
    transform=ccrs.PlateCarree(),
    scale=arrow_scale,         # adjust scaling of arrows
    width=0.005,       # adjust thickness
    headwidth=3
#    scale_units='xy'
    )



    ax4 = fig.add_subplot(3,4,i+3)

    diff_data = [
    diff_var_08_mag_core.values.flatten(),
    diff_var_06_mag_core.values.flatten(),
    diff_var_05_mag_core.values.flatten()
     ]
    
    diff_data_clean = [d[~np.isnan(d)] for d in diff_data]
    
    # Create box plot
    COLORS = ['darkblue', 'royalblue', 'lightsteelblue']
    violin_parts = ax4.violinplot(diff_data_clean, showmeans=True, showmedians=True, showextrema=False)
   
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
                'region': region_label,
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


    print(f"{'='*60}")

    print("\n\n" + "="*80)
    print("SUMMARY TABLE - All Regions")
    print("="*80)
    print(f"{'Region':<15} {'Resolution':<10} {'Mean':>8} {'Median':>8} {'Std':>8} {'Sample':>8}")
    print("-"*80)

    for stats in all_stats:
        print(f"{stats['region']:<15} {stats['resolution']:<10} "
          f"{stats['mean']:>8.2f} {stats['median']:>8.2f} "
          f"{stats['std']:>8.2f} {stats['sample_size']:>8}")
    print("="*80)

    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['10km', '40km', '80km'], fontweight='bold', fontsize=10)
    ax4.set_ylabel('IVT Difference',fontweight='bold', fontsize=10)
    ax4.set_title('ICON - ERA5',fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='grey', linestyle='-', linewidth=1.5, alpha=0.7)

# You might want to set consistent y-axis limits
    all_diffs = np.concatenate(diff_data_clean)
    ax4.set_ylim(-0.25,1.5)

#     if i == 9:
#           all_axes.extend([ax1, ax2, ax3, ax4])
#
#           cbar = fig.colorbar(all_axes[0].collections[0], ax=all_axes,
#                  pad=0.02, fraction=0.06,aspect=40,extend='both',orientation='horizontal')
#           cbar = ax1.collections[0].colorbar
#           cbar.set_label('Integrated Vertical Transport (Kg m-1 s-1)', fontsize=12, fontweight='bold')
#           cbar.ax.tick_params(axis='x', rotation=0, labelsize=10 )




# Apply your color scheme
    for pc, color in zip(violin_parts['bodies'], COLORS):
     pc.set_facecolor(color)
     pc.set_alpha(0.7)

# Customize the median and mean lines for better visibility
    violin_parts['cmedians'].set_color('black')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cmeans'].set_color('brown')
    violin_parts['cmeans'].set_linewidth(2)


    ax4.text(0.02, 0.98, core, transform=ax4.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))



# Add coast line 



    ax1.coastlines(resolution='10m',color='black',linewidth=2)
#    ax1.set_title('Average Monsoon Onset Date (Calendar Format - 10 Day Intervals)')
    
    ax1.set_title('ICON (10KM)',fontweight='bold', fontsize=14)
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
    ax2.set_title('ICON (40KM)',fontweight='bold', fontsize=14)
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
    ax3.set_title('ICON (80KM)',fontweight='bold', fontsize=14)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.xaxis.set_tick_params(labelsize=2)
    ax3.yaxis.set_tick_params(labelsize=2)
    ax3.set_aspect('auto')
    gls = ax3.gridlines(draw_labels=True,color="none")
    gls.top_labels=False
    gls.right_labels=False
#    _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)


    if i == 1:
            ax1.set_title('(a) ICON(10KM) - ERA5',fontweight='bold', fontsize=14)
            ax2.set_title('(b) ICON(40KM) - ERA5',fontweight='bold', fontsize=14)
            ax3.set_title('(c) ICON(80KM) - ERA5',fontweight='bold', fontsize=14)
            ax4.set_title('(d) Distribution of Biases',fontweight='bold', fontsize=14)

    if i == 5:
            ax1.set_title('(e) ICON(10KM) - ERA5',fontweight='bold', fontsize=14)
            ax2.set_title('(f) ICON(40KM) - ERA5',fontweight='bold', fontsize=14)
            ax3.set_title('(g) ICON(80KM) - ERA5',fontweight='bold', fontsize=14)
            ax4.set_title('(h) Distribution of Biases',fontweight='bold', fontsize=14)

    if i == 9:
            ax1.set_title('(i) ICON(10KM) - ERA5',fontweight='bold', fontsize=14)
            ax2.set_title('(j) ICON(40KM) - ERA5',fontweight='bold', fontsize=14)
            ax3.set_title('(k) ICON(80KM) - ERA5',fontweight='bold', fontsize=14)
            ax4.set_title('(l) Distribution of Biases',fontweight='bold', fontsize=14)


#plt.tight_layout()
plt.savefig(plot_name, dpi=600)
fig.savefig("Difference_IVT_Transport.pdf", bbox_inches='tight', pad_inches=0.1)
