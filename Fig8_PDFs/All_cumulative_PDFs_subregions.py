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

import cartopy.feature as cfeature

import matplotlib.colors as colors
##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

#Here choose NH_Summer or NH_Winter or NH_Summer-NH_Winter
season='NH_Summer'

# base directory where your analysis data is stored GPU
# Directory of 10KM simulation 

data_base_dir_08 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/tot_prec'
model_data_08    =  'tot_prec_30_day.nc'



# Directoy for 40KM simulation (Time step = 4018)
data_base_dir_06 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/tot_prec'
model_data_06    =  'tot_prec_30_day.nc'



# Directory for 80KM simulation (Time step = 4018)
data_base_dir_05 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/tot_prec'
model_data_05    =  'tot_prec_30_day.nc'


# Observational Data Sets Starts from Here. 

# # Observational Data Sets for IMERG (Time step = 4017)
data_base_dir_imerg = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/IMERG/day_nc4_files/post_processed_files'
model_data_imerg    =  'precipitation_cdo_all_rmp_10years.nc'


# Observational Data Sets for CPC (Time step = 4018)

data_base_dir_cpc = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/CPC_data/daily'
model_data_cpc    =  'precip.daily_rmp_10years.nc'

# Observational Data Sets for ERA5 (Time step = 4018)

data_base_dir_era5 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/ERA5_Data/daily'
model_data_era5    =  'pr_day_reanalysis_era5_r1i1p1_daily_rmp_10years.nc'


# Obesrvational Data Set for MSWEP (Time step = 4018)

data_base_dir_mswep = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/MSWEP/Data_2006_2022'
model_data_mswep    = 'precip_rmp_10years.nc'


# CMORPH Data ((Time step = 4018))

data_base_dir_cmorph = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/CMORPH/data/CMORPH/daily_nc'
model_data_cmorph    = 'daily_rmp_10years.nc'

# GPCC Data ((Time step = 4018) 

data_base_dir_gpcc = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/GPCC_data/daily'
model_data_gpcc    = 'full_data_rmp_10years.nc'



# netCDF variable name of variable to plot
var='tot_prec','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_imerg    = os.path.join(data_base_dir_imerg, model_data_imerg)
model_data_dir_cpc    = os.path.join(data_base_dir_cpc, model_data_cpc)
model_data_dir_era5    = os.path.join(data_base_dir_era5, model_data_era5)
model_data_dir_mswep    = os.path.join(data_base_dir_mswep, model_data_mswep)
model_data_dir_cmorph    = os.path.join(data_base_dir_cmorph, model_data_cmorph)
model_data_dir_gpcc    = os.path.join(data_base_dir_gpcc, model_data_gpcc)


ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_imerg  = xr.open_dataset(model_data_dir_imerg)
ds_cpc  = xr.open_dataset(model_data_dir_cpc)
ds_era5  = xr.open_dataset(model_data_dir_era5)
ds_mswep  = xr.open_dataset(model_data_dir_mswep)
ds_cmorph  = xr.open_dataset(model_data_dir_cmorph)
ds_gpcc  = xr.open_dataset(model_data_dir_gpcc)


plot_var_key = 'tot_prec'
# Read the Variable from Here.  
#var_08 = ds_08[plot_var_key]
#var_06 = ds_06[plot_var_key]
#var_05 = ds_05[plot_var_key] 
#var_imerg = ds_imerg['precipitation']
#var_cpc = ds_cpc['precip']
#var_era5 = ds_era5['pr']
#var_mswep = ds_mswep['precipitation']
#var_cmorph = ds_cmorph['cmorph']
#var_gpcc = ds_gpcc['precip']


# Special Operationfor IMERG

#var_imerg = var_imerg.transpose()

reg = 'South_Asia','North Africa','North America', 'North Asia', 'NS Africa', 'NN America'

#reg = 'South_Asia', 'South_Asia', 'South_Asia'

fig = plt.figure(figsize=(12, 8), constrained_layout=True)

for x in reg:

     if x  == 'South_Asia':
    # Select a Region
    #South Asia
          lat_min=18
          lat_max=25
          lon_min=72
          lon_max=85
          plot_name='South Asia'
          plot_name = '(a) SAsiaM (Indian Core)'
          k = 1

     elif x  == 'North Africa':
          lat_min=10
          lat_max=20
          lon_min=-18
          lon_max=16
          plot_name='Sahel region'
          plot_name = '(b) WAfriM (African Core)'
          k = 2

     elif x == 'NN America':
          lat_min=20
          lat_max=30
          lon_min=-110
          lon_max=-100
          plot_name='North America'
          plot_name = '(c) NAmerM (N.American Core)'
          k = 3

     elif x  == 'NS Africa':
          lat_min=10
          lat_max=20
          lon_min=-18
          lon_max=16
          plot_name='Sahel region'
          plot_name = '(e) WAfriM (African Core)'
          k = 5

     elif x == 'North America':
          lat_min=20
          lat_max=30
          lon_min=-110
          lon_max=-100
          plot_name='North America'
          plot_name = '(f) NAmerM (N.American Core)'
          k = 6

     elif x  == 'North Asia':
    # Select a Region
    #South Asia
          lat_min=18
          lat_max=25
          lon_min=72
          lon_max=85
          plot_name='South Asia'
          plot_name = '(d) SAsiaM (Indian Core)'
          k = 4



     var_08 = ds_08[plot_var_key]
     var_06 = ds_06[plot_var_key]
     var_05 = ds_05[plot_var_key]
     var_imerg = ds_imerg['precipitation']
     var_cpc = ds_cpc['precip']
     var_era5 = ds_era5['pr']
     var_mswep = ds_mswep['precipitation']
     var_cmorph = ds_cmorph['cmorph']
     var_gpcc = ds_gpcc['precip']


     var_imerg = var_imerg.transpose()

# Select Lat & Lon focussing on the tropics 
     var_08 = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_06 = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_05 = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_imerg = var_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_cpc = var_cpc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_era5 = var_era5.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_mswep = var_mswep.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_cmorph = var_cmorph.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
     var_gpcc = var_gpcc.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

# We calculate Only Norther Hemisphere Interannual Variability

     var_08_summer = var_08.sel(time=var_08.time.dt.month.isin([6, 7, 8, 9]))

     var_06_summer = var_06.sel(time=var_06.time.dt.month.isin([6, 7, 8, 9]))

     var_05_summer = var_05.sel(time=var_05.time.dt.month.isin([6, 7, 8, 9]))

     var_imerg_summer = var_imerg.sel(time=var_imerg.time.dt.month.isin([6, 7, 8, 9]))

     var_mswep_summer = var_mswep.sel(time=var_mswep.time.dt.month.isin([6, 7, 8, 9]))

     var_cmorph_summer = var_cmorph.sel(time=var_cmorph.time.dt.month.isin([6, 7, 8, 9]))



#Convert the Data Here into Array. 

     var_08_summer_numpy = var_08_summer.to_numpy()
     var_08_summer_numpy_vector = var_08_summer_numpy.reshape(-1) 
     var_08_summer_numpy_vector_rain=var_08_summer_numpy_vector[var_08_summer_numpy_vector > 0.1]


     var_06_summer_numpy = var_06_summer.to_numpy()
     var_06_summer_numpy_vector = var_06_summer_numpy.reshape(-1)
     var_06_summer_numpy_vector_rain=var_06_summer_numpy_vector[var_06_summer_numpy_vector > 0.1]


     var_05_summer_numpy = var_05_summer.to_numpy()
     var_05_summer_numpy_vector = var_05_summer_numpy.reshape(-1)
     var_05_summer_numpy_vector_rain=var_05_summer_numpy_vector[var_05_summer_numpy_vector > 0.1]


     var_imerg_summer_numpy = var_imerg_summer.to_numpy()
     var_imerg_summer_numpy_vector = var_imerg_summer_numpy.reshape(-1)
     var_imerg_summer_numpy_vector_rain=var_imerg_summer_numpy_vector[var_imerg_summer_numpy_vector > 0.1]


     var_mswep_summer_numpy = var_mswep_summer.to_numpy()
     var_mswep_summer_numpy_vector = var_mswep_summer_numpy.reshape(-1)
     var_mswep_summer_numpy_vector_rain=var_mswep_summer_numpy_vector[var_mswep_summer_numpy_vector > 0.1]


     var_cmorph_summer_numpy = var_cmorph_summer.to_numpy()
     var_cmorph_summer_numpy_vector = var_cmorph_summer_numpy.reshape(-1)
     var_cmorph_summer_numpy_vector_rain=var_cmorph_summer_numpy_vector[var_cmorph_summer_numpy_vector > 0.1]




#def plot_loghist(x, bins):
#  hist, bins = np.histogram(x, bins=bins)
#  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#  plt.hist(x, bins=logbins,density=True)
#  plt.xscale('log')
#  plt.yscale('log')

#plot_loghist(var_08_summer_numpy_vector_rain, 30)

##   Here draw solid lines for the bins. 

     data = [var_08_summer_numpy_vector_rain,var_06_summer_numpy_vector_rain,var_05_summer_numpy_vector_rain,var_mswep_summer_numpy_vector_rain,var_imerg_summer_numpy_vector_rain]

     labels=["ICON (10KM)", "ICON (40KM)", "ICON (80KM)","MSWEP (OBS)", "IMERG (OBS)"]

     colors = ['r','b','g','k','k']
     #bins=np.histogram(np.hstack((var_08_summer_numpy_vector_rain,var_06_summer_numpy_vector_rain,var_05_summer_numpy_vector_rain,var_mswep_summer_numpy_vector_rain,var_imerg_summer_numpy_vector_rain)), bins=20)[1]
#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

     min_val = 0.1    # 0.1 mm/day
     max_val = 500   # 100 mm/day (or adjust based on your data range)
     bins = np.logspace(np.log10(min_val), np.log10(max_val), 20)


     linestyle = ['-','-','-','-','--']

     ax = fig.add_subplot(2,3,k)

     for i in range(len(data)):
              print(i)
              print(f"  Plotting dataset {i} in subplot {k}")
#              hist, bins = np.histogram(data[i], bins=bins,density=True) 
#              ax.plot(bins[0:len(bins)-1],hist,c=colors[i],label=labels[i], linestyle=linestyle[i],linewidth=2.0)
               # Calculate histogram (counts, not density)
              hist_counts, bin_edges = np.histogram(data[i], bins=bins)
    
               # Calculate CDF: cumulative sum of counts, normalized by total samples
              cdf = np.cumsum(hist_counts) / len(data[i])
    
    # Plot CDF at the bin edges (shift to right edge for standard CDF plot)
              
              if k == 1 or k == 2 or k == 3: 
                        ax.plot(bin_edges[1:], cdf, c=colors[i], label=labels[i], linestyle=linestyle[i], linewidth=2.0)
              if k == 4 or k == 5 or k == 6: 
                        ax.plot(bin_edges[1:], 1-cdf, c=colors[i], label=labels[i], linestyle=linestyle[i], linewidth=2.0)


     # CORRECTED LINES:
     
     if k == 1 or k == 2 or k == 3:
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.set_ylim(0, 1.05) 

      
     if k == 4 or k == 5 or k == 6:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1)  # Set lower limit to avoid log(0) issues
        #ax.set_xlim(1,500)



     if k == 3: 
        ax.legend()
 #    ax.set_ylim(0.001, 1)
 #    ax.set_xlabel("Precipitation (mm/day)", fontweight='bold', fontsize=12)
     if k == 1: 
        ax.set_ylabel("Cumulative Probability", fontweight='bold', fontsize=12)
     ax.set_title(plot_name, fontweight='bold', fontsize=12)

     if k == 4:
        ax.set_ylabel("Probability of Exceedance", fontweight='bold', fontsize=12)
     ax.set_title(plot_name, fontweight='bold', fontsize=12)

     if k == 4 or k == 5 or k == 6:
        ax.set_xlabel("Total Precipitation (mm/day)", fontweight='bold', fontsize=12)




     pos = ax.get_position()

# Create inset relative to main subplot position
#     inset_left = pos.x0 + 0.02    # 2% from left edge of subplot
#     inset_bottom = pos.y0 + 0.02  # 2% from bottom edge of subplot  
#     inset_width = pos.width * 0.5  # 50% of subplot width
#     inset_height = pos.height * 0.5  # 50% of subplot height
#     inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height], 
#                       projection=ccrs.PlateCarree())


     if k ==1:  
        inset_left = (k-1)/3 + 0.009  # Manual positioning based on subplot index
        inset_bottom = 0.2  # Fixed bottom position
        inset_width  = 0.2   # Fixed width
        inset_height = 0.2  # Fixed height

        inset_left = pos.x0 + 0.06 
        inset_bottom = pos.y0 + 0.03
        inset_width = pos.width * 0.45
        inset_height = pos.height * 0.45




     if k ==2: 
        inset_left = (k-1)/3 + 0.09  # Manual positioning based on subplot index
        inset_bottom = 0.2  # Fixed bottom position
        inset_width  = 0.2   # Fixed width
        inset_height = 0.2  # Fixed height

        inset_left = pos.x0 + 0.1
        inset_bottom = pos.y0 + 0.02
        inset_width = pos.width * 0.65
        inset_height = pos.height * 0.65




     if k ==3: 
        inset_left = (k-1)/3 + 1.1  # Manual positioning based on subplot index
        inset_bottom = 0.45  # Fixed bottom position
        inset_width  = 0.2   # Fixed width
        inset_height = 0.2  # Fixed height

        inset_left = pos.x0 + 0.21
        inset_bottom = pos.y0 + 0.15
        inset_width = pos.width * 0.45
        inset_height = pos.height * 0.45



     if k ==4:
        inset_left = (k-1)/3 + 0.01  # Manual positioning based on subplot index
        inset_bottom = 0.45  # Fixed bottom position
        inset_width  = 0.2   # Fixed width
        inset_height = 0.2  # Fixed height


        inset_left = pos.x0 - 0.02
        inset_bottom = pos.y0 - 0.02
        inset_width = pos.width * 0.45
        inset_height = pos.height * 0.45


     if k ==5:
        inset_left = (k-1)/3 - 0.01  # Manual positioning based on subplot index
        inset_bottom = 0.45  # Fixed bottom position
        inset_width  = 0.2   # Fixed width
        inset_height = 0.2  # Fixed height 


        inset_left = pos.x0 + 0.02
        inset_bottom = pos.y0 - 0.04
        inset_width = pos.width * 0.65
        inset_height = pos.height * 0.65






     if k ==6:
        inset_left = (k-1)/3 + 0.21  # Manual positioning based on subplot index
        inset_bottom = 0.45  # Fixed bottom position
        inset_width  = 0.2   # Fixed width
        inset_height = 0.2  # Fixed height


        inset_left = pos.x0 + 0.06
        inset_bottom = pos.y0 - 0.04
        inset_width = pos.width * 0.45
        inset_height = pos.height * 0.45





     inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height],
                       projection=ccrs.PlateCarree())


     
     if x  == 'South_Asia' or 'North Asia':
    # South Asia coordinates
        lat_min, lat_max = 18, 25
        lon_min, lon_max = 72, 85
        plot_name = 'SAsiaM (Indian Core)'
    
    # Create inset map in bottom left corner
        inset_ax.set_extent([60, 110, 0, 40], crs=ccrs.PlateCarree())
        inset_ax.add_feature(cfeature.COASTLINE)
        inset_ax.add_feature(cfeature.BORDERS)
        inset_ax.add_feature(cfeature.LAND, color='lightgray')
        inset_ax.add_feature(cfeature.OCEAN, color='lightblue')
    
    # Add rectangle to highlight the region
        inset_ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                   fill=False, edgecolor='red', linewidth=2, 
                                   transform=ccrs.PlateCarree()))


#     if x  == 'North Asia':
    # South Asia coordinates
#        lat_min, lat_max = 25, 35
#        lon_min, lon_max = 72, 95
#        plot_name = 'SAsiaM (Himalayan)'

    # Create inset map in bottom left corner
#        inset_ax.set_extent([60, 110, 0, 40], crs=ccrs.PlateCarree())
#        inset_ax.add_feature(cfeature.COASTLINE)
#        inset_ax.add_feature(cfeature.BORDERS)
#        inset_ax.add_feature(cfeature.LAND, color='lightgray')
#        inset_ax.add_feature(cfeature.OCEAN, color='lightblue')
#
    # Add rectangle to highlight the region
#        inset_ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
#                                   fill=False, edgecolor='red', linewidth=2,
#                                   transform=ccrs.PlateCarree()))




     if x == 'NS Africa' or x == 'North Africa'  :
    # North Africa coordinates
        lat_min, lat_max = 10, 20
        lon_min, lon_max = -18, 16
        plot_name = 'WAfriM (Sahel Region)'
    
    # Create inset map
        inset_ax.set_extent([-65, 65, -10, 30], crs=ccrs.PlateCarree())  # Wider view
        inset_ax.add_feature(cfeature.COASTLINE)
        inset_ax.add_feature(cfeature.BORDERS)
        inset_ax.add_feature(cfeature.LAND, color='lightgray')
        inset_ax.add_feature(cfeature.OCEAN, color='lightblue')
    
    # Highlight region
        inset_ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                   fill=False, edgecolor='red', linewidth=2,
                                   transform=ccrs.PlateCarree()))



#     if x == 'NS Africa':
    # North Africa coordinates
#        lat_min, lat_max = 5, 10
#        lon_min, lon_max = -18, 16
#        plot_name = 'WAfriM (Guiena Coast)'

    # Create inset map
#        inset_ax.set_extent([-65, 65, -10, 30], crs=ccrs.PlateCarree())  # Wider view
#        inset_ax.add_feature(cfeature.COASTLINE)
#        inset_ax.add_feature(cfeature.BORDERS)
#        inset_ax.add_feature(cfeature.LAND, color='lightgray')
#        inset_ax.add_feature(cfeature.OCEAN, color='lightblue')

    # Highlight region
#        inset_ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
#                                   fill=False, edgecolor='red', linewidth=2,
#                                   transform=ccrs.PlateCarree()))



     if x == 'NN America' or x == 'North America':
    # South America coordinates
        lat_min, lat_max = 20, 30
        lon_min, lon_max = -110, -100
        plot_name = 'NAmerM (Mexican Coast)'
    # Create inset map
        inset_ax.set_extent([-140, -75, 0, 40], crs=ccrs.PlateCarree())
        inset_ax.add_feature(cfeature.COASTLINE)
        inset_ax.add_feature(cfeature.BORDERS)
        inset_ax.add_feature(cfeature.LAND, color='lightgray')
        inset_ax.add_feature(cfeature.OCEAN, color='lightblue')
    
    # Highlight region
        inset_ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                   fill=False, edgecolor='red', linewidth=2,
                                   transform=ccrs.PlateCarree()))



#     if x == 'NN America':
#    # South America coordinates
#        lat_min, lat_max = 20, 30
#        lon_min, lon_max = -110, -100
#        plot_name = 'NAmerM (Continental US)'

    # Create inset map
#        inset_ax.set_extent([-140, -75, 0, 40], crs=ccrs.PlateCarree())
#        inset_ax.add_feature(cfeature.COASTLINE)
#        inset_ax.add_feature(cfeature.BORDERS)
#        inset_ax.add_feature(cfeature.LAND, color='lightgray')
#        inset_ax.add_feature(cfeature.OCEAN, color='lightblue')

    # Highlight region
#        inset_ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
#                                   fill=False, edgecolor='red', linewidth=2,
#                                   transform=ccrs.PlateCarree()))



fig.savefig("Fig8_cumulative_PDF_plot_sub.pdf", bbox_inches='tight', pad_inches=0.1)
