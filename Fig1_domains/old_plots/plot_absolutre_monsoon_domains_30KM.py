#!/capstor/store/cscs/exclaim/excp01/ppothapa/.env_icon/bin/python
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

from scipy.stats import pearsonr

##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

#Here choose NH_Summer or NH_Winter or NH_Summer-NH_Winter
season='NH_Summer'



def calculate_spatial_correlation(model_data, obs_data):
    """
    Helper function to calculate spatial correlation between two DataArrays.
    Returns correlation coefficient and p-value.
    """
    # Flatten both arrays to 1D
    model_flat = model_data.values.flatten()
    obs_flat = obs_data.values.flatten()

    # Create mask where both arrays have valid data (non-NaN)
    valid_mask = ~(np.isnan(model_flat) | np.isnan(obs_flat))

    # Apply the mask
    model_valid = model_flat[valid_mask]
    obs_valid = obs_flat[valid_mask]

    # Check if we have enough points for correlation
    if len(model_valid) < 2:
        print(f"Warning: Only {len(model_valid)} valid points for correlation")
        return np.nan, np.nan

    # Calculate correlation
    return pearsonr(model_valid, obs_valid)


def calculate_spatial_rmse(model_data, obs_data):
    """
    Helper function to calculate Root-Mean-Square Error (RMSE) between two DataArrays.
    Returns RMSE value.
    """
    # Flatten both arrays to 1D
    model_flat = model_data.values.flatten()
    obs_flat = obs_data.values.flatten()
    
    # Create mask where both arrays have valid data (non-NaN)
    valid_mask = ~(np.isnan(model_flat) | np.isnan(obs_flat))
    
    # Apply the mask
    model_valid = model_flat[valid_mask]
    obs_valid = obs_flat[valid_mask]
    
    # Check if we have enough points
    if len(model_valid) < 2:
        print(f"Warning: Only {len(model_valid)} valid points for RMSE")
        return np.nan
    
    # Calculate RMSE: sqrt(mean((model - obs)^2))
    squared_errors = (model_valid - obs_valid) ** 2
    mse = np.mean(squared_errors)  # Mean Squared Error
    rmse = np.sqrt(mse)            # Root Mean Squared Error
    
    return rmse







# base directory where your analysis data is stored GPU
# Directory of 10KM simulation
data_base_dir_08 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/tot_prec'
model_data_08    =  'tot_prec_30_day.nc'

# Directoy for 40KM simulation (Time step = 4018)
data_base_dir_06 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/tot_prec'
model_data_06    =  'tot_prec_30_day.nc'

# Directory for 80KM simulation (Time step = 4018)
data_base_dir_05 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/tot_prec'
model_data_05    =  'tot_prec_30_day.nc'

# Observational Data Sets Starts from Here.

# # Observational Data Sets for IMERG (Time step = 3653)
data_base_dir_imerg = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/IMERG/day_nc4_files/post_processed_files'
model_data_imerg    =  'precipitation_cdo_all_rmp_10years.nc'

# Observational Data Sets for CPC (Time step = 3653)

data_base_dir_cpc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CPC_data/daily'
model_data_cpc    =  'precip.daily_rmp_10years.nc'

# Observational Data Sets for ERA5 (Time step = 3653)

data_base_dir_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/daily'
model_data_era5    =  'pr_day_reanalysis_era5_r1i1p1_daily_rmp_10years.nc'

# Obesrvational Data Set for MSWEP (Time step = 3653)

data_base_dir_mswep = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/MSWEP/Data_2006_2022'
model_data_mswep    = 'precip_rmp_10years.nc'

# CMORPH Data ((Time step = 3653))

data_base_dir_cmorph = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CMORPH/data/CMORPH/daily_nc'
model_data_cmorph    = 'daily_rmp_10years.nc'

# GPCC Data ((Time step = 3653)

data_base_dir_gpcc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/GPCC_data/daily'
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



# Color Scales Range
a = [0,.01,.1,.25,.5,1,1.5,2,3,4,6,8,10,15,20,30]
a = [0,.005,.01,.15,.25,0.5,1,2,3,4,5,6,7,8,20,30]
a = [-30,-20,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,20,30]


# Normalize the bin between 0 and 1 (uneven bins are important here)
norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]

# Color tuple for every bin
C = np.array([[255,255,255],
              [199,233,192],
              [161,217,155],
              [116,196,118],
              [49,163,83],
              [0,109,44],
              [255,250,138],
              [255,204,79],
              [254,141,60],
              [252,78,42],
              [214,26,28],
              [173,0,38],
              [112,0,38],
              [59,0,48],
              [76,0,115],
              [255,219,255]])

# Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
COLORS = []
for i, n in enumerate(norm):
    COLORS.append((n, np.array(C[i])/255.))

# Create the colormap
cmap = colors.LinearSegmentedColormap.from_list("precipitation", COLORS)

bounds = np.linspace(55,100,5)
bounds = np.arange(0,16.1,1) # Here the limit is till 55 to 100, it is strange in python
if season=='NH_Summer-NH_Winter':
    bounds = np.arange(-10,15,5)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)


# The plotting call from here 
for x in var:
# Plot Name and Name of the Variable for Plotting.
     plot_var_key = x
     print(plot_var_key)
     plot_name = x
     print(plot_name)

# Read the Variable from Here.  
     var_08 = ds_08[plot_var_key]
     var_06 = ds_06[plot_var_key]
     var_05 = ds_05[plot_var_key] 
     var_imerg = ds_imerg['precipitation']
     var_cpc = ds_cpc['precip']
     var_era5 = ds_era5['pr']
     var_mswep = ds_mswep['precipitation']
     var_cmorph = ds_cmorph['cmorph']
     var_gpcc = ds_gpcc['precip']


# Special Operationfor IMERG

     var_imerg = var_imerg.transpose()

# Select Lat & Lon focussing on the tropics 
     var_08 = var_08.sel(lat=slice(-40,40))
     var_06 = var_06.sel(lat=slice(-40,40))
     var_05 = var_05.sel(lat=slice(-40,40))
     var_imerg = var_imerg.sel(lat=slice(-40,40))
     var_cpc = var_cpc.sel(lat=slice(-40,40))
     var_era5 = var_era5.sel(lat=slice(-40,40))
     var_mswep = var_mswep.sel(lat=slice(-40,40))
     var_cmorph = var_cmorph.sel(lat=slice(-40,40))
     var_gpcc = var_gpcc.sel(lat=slice(-40,40))


# Select the Seasons and Mean over the Seasons.      
     var_08_summer = var_08.sel(time=var_08.time.dt.month.isin([6, 7, 8, 9]))
     var_08_winter = var_08.sel(time=var_08.time.dt.month.isin([12, 1, 2, 3]))

     var_06_summer = var_06.sel(time=var_06.time.dt.month.isin([6, 7, 8, 9]))
     var_06_winter = var_06.sel(time=var_06.time.dt.month.isin([12, 1, 2, 3]))

     var_05_summer = var_05.sel(time=var_05.time.dt.month.isin([6, 7, 8, 9]))
     var_05_winter = var_05.sel(time=var_05.time.dt.month.isin([12, 1, 2, 3]))
    
     var_imerg_summer = var_imerg.sel(time=var_imerg.time.dt.month.isin([6, 7, 8, 9]))
     var_imerg_winter = var_imerg.sel(time=var_imerg.time.dt.month.isin([12, 1, 2, 3]))

     var_era5_summer = var_era5.sel(time=var_era5.time.dt.month.isin([6, 7, 8, 9]))
     var_era5_winter = var_era5.sel(time=var_era5.time.dt.month.isin([12, 1, 2, 3]))

     var_cpc_summer = var_cpc.sel(time=var_cpc.time.dt.month.isin([6, 7, 8, 9]))
     var_cpc_winter = var_cpc.sel(time=var_cpc.time.dt.month.isin([12, 1, 2, 3]))

     var_mswep_summer = var_mswep.sel(time=var_mswep.time.dt.month.isin([6, 7, 8, 9]))
     var_mswep_winter = var_mswep.sel(time=var_mswep.time.dt.month.isin([12, 1, 2, 3]))

     var_cmorph_summer = var_cmorph.sel(time=var_cmorph.time.dt.month.isin([6, 7, 8, 9]))
     var_cmorph_winter = var_cmorph.sel(time=var_cmorph.time.dt.month.isin([12, 1, 2, 3]))

     var_gpcc_summer = var_gpcc.sel(time=var_gpcc.time.dt.month.isin([6, 7, 8, 9]))
     var_gpcc_winter = var_gpcc.sel(time=var_gpcc.time.dt.month.isin([12, 1, 2, 3]))     


#   Here calculate the summer - winter (mm/day) for the plot

     var_08_summer_winter = var_08_summer.mean("time") - var_08_winter.mean("time")
     var_06_summer_winter = var_06_summer.mean("time") - var_06_winter.mean("time")
     var_05_summer_winter = var_05_summer.mean("time") - var_05_winter.mean("time")
     var_era5_summer_winter =  var_era5_summer.mean("time")*86400 - var_era5_winter.mean("time")*86400 
     var_imerg_summer_winter = var_imerg_summer.mean("time") - var_imerg_winter.mean("time")
     var_cpc_summer_winter = var_cpc_summer.mean("time") - var_cpc_winter.mean("time")
     var_mswep_summer_winter = var_mswep_summer.mean("time") - var_mswep_winter.mean("time") 
     var_cmorph_summer_winter = var_cmorph_summer.mean("time") - var_cmorph_winter.mean("time")
     var_gpcc_summer_winter = var_gpcc_summer.mean("time") - var_gpcc_winter.mean("time")


#    Store the mask files. 
     ds_08_mask     = xr.open_dataset("08_mask_2mm_55_na.nc")
     ds_06_mask     = xr.open_dataset("06_mask_2mm_55_na.nc")
     ds_05_mask     = xr.open_dataset("05_mask_2mm_55_na.nc")
     ds_era5_mask   = xr.open_dataset("era5_mask_2mm_55_na.nc")
     ds_imerg_mask  = xr.open_dataset("imerg_mask_2mm_55_na.nc")
     ds_cpc_mask    = xr.open_dataset("cpc_mask_2mm_55_na.nc")
     ds_mswep_mask  = xr.open_dataset("mswep_mask_2mm_55_na.nc")
     ds_cmorph_mask = xr.open_dataset("cmorph_mask_2mm_55_na.nc")
     ds_gpcc_mask   = xr.open_dataset("gpcc_mask_2mm_55_na.nc")

#    Variable Mask 

     var_08_mask    = ds_08_mask["__xarray_dataarray_variable__"]
     var_06_mask    = ds_06_mask["__xarray_dataarray_variable__"]
     var_05_mask    = ds_05_mask["__xarray_dataarray_variable__"]
     var_era5_mask  = ds_era5_mask["__xarray_dataarray_variable__"]
     var_imerg_mask = ds_imerg_mask["__xarray_dataarray_variable__"]
     var_cpc_mask   = ds_cpc_mask["__xarray_dataarray_variable__"]
     var_mswep_mask = ds_mswep_mask["__xarray_dataarray_variable__"]
     var_cmorph_mask = ds_cmorph_mask["__xarray_dataarray_variable__"]
     var_gpcc_mask  = ds_gpcc_mask["__xarray_dataarray_variable__"]


##   Magnitude of the Precipitation
     var_08_summer_winter_mag  = abs(var_08_summer_winter * var_08_mask)
     var_06_summer_winter_mag  = abs(var_06_summer_winter * var_06_mask)
     var_05_summer_winter_mag  = abs(var_05_summer_winter * var_05_mask)
     var_era5_summer_winter_mag = abs(var_era5_summer_winter * var_era5_mask)
     var_imerg_summer_winter_mag = abs(var_imerg_summer_winter * var_imerg_mask)
     var_cpc_summer_winter_mag   = abs(var_cpc_summer_winter * var_cpc_mask)
     var_mswep_summer_winter_mag = abs(var_mswep_summer_winter * var_mswep_mask)
     var_cmorph_summer_winter_mag = abs(var_cmorph_summer_winter * var_cmorph_mask)
     var_gpcc_summer_winter_mag   = abs(var_gpcc_summer_winter * var_gpcc_mask)


## Filtering values for the countours: 

     var_08_summer_winter_NH     = var_08_summer_winter.where((var_08_summer_winter > 0) & (var_08_summer_winter.lat > -10),np.nan)
     var_08_summer_winter_SH     = var_08_summer_winter.where((var_08_summer_winter < 0) & (var_08_summer_winter.lat < 10), np.nan)
     
     var_06_summer_winter_NH     = var_06_summer_winter.where((var_06_summer_winter > 0) & (var_06_summer_winter.lat > -10),np.nan) 
     var_06_summer_winter_SH     = var_06_summer_winter.where((var_06_summer_winter < 0) & (var_06_summer_winter.lat < 10), np.nan)

     var_05_summer_winter_NH     = var_05_summer_winter.where((var_05_summer_winter > 0) & (var_05_summer_winter.lat > -10),np.nan)
     var_05_summer_winter_SH     = var_05_summer_winter.where((var_05_summer_winter < 0) & (var_05_summer_winter.lat < 10), np.nan)
 
     var_era5_summer_winter_NH   = var_era5_summer_winter.where((var_era5_summer_winter > 0) & (var_era5_summer_winter.lat > -10),np.nan) 
     var_era5_summer_winter_SH   = var_era5_summer_winter.where((var_era5_summer_winter < 0) & (var_era5_summer_winter.lat < 10), np.nan)   

     var_imerg_summer_winter_NH  = var_imerg_summer_winter.where((var_imerg_summer_winter > 0) & (var_imerg_summer_winter.lat > -10),np.nan)
     var_imerg_summer_winter_SH  = var_imerg_summer_winter.where((var_imerg_summer_winter < 0) & (var_imerg_summer_winter.lat < 10), np.nan)    

     var_cpc_summer_winter_NH    = var_cpc_summer_winter.where((var_cpc_summer_winter > 0) & (var_cpc_summer_winter.lat > -10),np.nan)
     var_cpc_summer_winter_SH    = var_cpc_summer_winter.where((var_cpc_summer_winter < 0) & (var_cpc_summer_winter.lat < 10), np.nan)

     var_mswep_summer_winter_NH  = var_mswep_summer_winter.where((var_mswep_summer_winter > 0) & (var_mswep_summer_winter.lat > -10),np.nan)
     var_mswep_summer_winter_SH  = var_mswep_summer_winter.where((var_mswep_summer_winter < 0) & (var_mswep_summer_winter.lat < 10), np.nan)    

     var_cmorph_summer_winter_NH = var_cmorph_summer_winter.where((var_cmorph_summer_winter > 0) & (var_cmorph_summer_winter.lat > -10),np.nan)
     var_cmorph_summer_winter_SH = var_cmorph_summer_winter.where((var_cmorph_summer_winter < 0) & (var_cmorph_summer_winter.lat < 10), np.nan)    

     var_gpcc_summer_winter_NH   = var_gpcc_summer_winter.where((var_gpcc_summer_winter > 0) & (var_gpcc_summer_winter.lat > -10),np.nan)
     var_gpcc_summer_winter_SH   = var_gpcc_summer_winter.where((var_gpcc_summer_winter < 0) & (var_gpcc_summer_winter.lat < 10), np.nan)




     # Calculate correlations for IMERG
     corr_imerg_08, p_value_imerg_08 = calculate_spatial_correlation(var_08_summer_winter_mag, var_imerg_summer_winter_mag)
     corr_imerg_06, p_value_imerg_06 = calculate_spatial_correlation(var_06_summer_winter_mag, var_imerg_summer_winter_mag)
     corr_imerg_05, p_value_imerg_05 = calculate_spatial_correlation(var_05_summer_winter_mag, var_imerg_summer_winter_mag)

     rmse_imerg_08  = calculate_spatial_rmse(var_08_summer_winter_mag, var_imerg_summer_winter_mag)
     rmse_imerg_06  = calculate_spatial_rmse(var_06_summer_winter_mag, var_imerg_summer_winter_mag)
     rmse_imerg_05  = calculate_spatial_rmse(var_05_summer_winter_mag, var_imerg_summer_winter_mag)

     # Calculate correlations for MSWEP
     corr_mswep_08, p_value_mswep_08 = calculate_spatial_correlation(var_08_summer_winter_mag, var_mswep_summer_winter_mag)
     corr_mswep_06, p_value_mswep_06 = calculate_spatial_correlation(var_06_summer_winter_mag, var_mswep_summer_winter_mag)
     corr_mswep_05, p_value_mswep_05 = calculate_spatial_correlation(var_05_summer_winter_mag, var_mswep_summer_winter_mag)

     rmse_mswep_08  = calculate_spatial_rmse(var_08_summer_winter_mag, var_mswep_summer_winter_mag)
     rmse_mswep_06  = calculate_spatial_rmse(var_06_summer_winter_mag, var_mswep_summer_winter_mag)
     rmse_mswep_05  = calculate_spatial_rmse(var_05_summer_winter_mag, var_mswep_summer_winter_mag)

      # Calulate for ERA5

     corr_era5_08, p_value_era5_08 = calculate_spatial_correlation(var_08_summer_winter_mag, var_era5_summer_winter_mag)
     corr_era5_06, p_value_era5_06 = calculate_spatial_correlation(var_06_summer_winter_mag, var_era5_summer_winter_mag)
     corr_era5_05, p_value_era5_05 = calculate_spatial_correlation(var_05_summer_winter_mag, var_era5_summer_winter_mag)

     rmse_era5_08  = calculate_spatial_rmse(var_08_summer_winter_mag, var_era5_summer_winter_mag)
     rmse_era5_06  = calculate_spatial_rmse(var_06_summer_winter_mag, var_era5_summer_winter_mag)
     rmse_era5_05  = calculate_spatial_rmse(var_05_summer_winter_mag, var_era5_summer_winter_mag)


     # Calculate for CMORPH

     corr_cmorph_08, p_value_cmorph_08 = calculate_spatial_correlation(var_08_summer_winter_mag, var_cmorph_summer_winter_mag)
     corr_cmorph_06, p_value_cmorph_06 = calculate_spatial_correlation(var_06_summer_winter_mag, var_cmorph_summer_winter_mag)
     corr_cmorph_05, p_value_cmorph_05 = calculate_spatial_correlation(var_05_summer_winter_mag, var_cmorph_summer_winter_mag)

     rmse_cmorph_08  = calculate_spatial_rmse(var_08_summer_winter_mag, var_cmorph_summer_winter_mag)
     rmse_cmorph_06  = calculate_spatial_rmse(var_06_summer_winter_mag, var_cmorph_summer_winter_mag)
     rmse_cmorph_05  = calculate_spatial_rmse(var_05_summer_winter_mag, var_cmorph_summer_winter_mag)


     # Calculate for CPC

     corr_cpc_08, p_value_cpc_08 = calculate_spatial_correlation(var_08_summer_winter_mag, var_cpc_summer_winter_mag)
     corr_cpc_06, p_value_cpc_06 = calculate_spatial_correlation(var_06_summer_winter_mag, var_cpc_summer_winter_mag)
     corr_cpc_05, p_value_cpc_05 = calculate_spatial_correlation(var_05_summer_winter_mag, var_cpc_summer_winter_mag)

     rmse_cpc_08  = calculate_spatial_rmse(var_08_summer_winter_mag, var_cpc_summer_winter_mag)
     rmse_cpc_06  = calculate_spatial_rmse(var_06_summer_winter_mag, var_cpc_summer_winter_mag)
     rmse_cpc_05  = calculate_spatial_rmse(var_05_summer_winter_mag, var_cpc_summer_winter_mag)


     # Calculöate for GPCC

     corr_gpcc_08, p_value_gpcc_08 = calculate_spatial_correlation(var_08_summer_winter_mag, var_gpcc_summer_winter_mag)
     corr_gpcc_06, p_value_gpcc_06 = calculate_spatial_correlation(var_06_summer_winter_mag, var_gpcc_summer_winter_mag)
     corr_gpcc_05, p_value_gpcc_05 = calculate_spatial_correlation(var_05_summer_winter_mag, var_gpcc_summer_winter_mag)
 
     rmse_gpcc_08  = calculate_spatial_rmse(var_08_summer_winter_mag, var_gpcc_summer_winter_mag)
     rmse_gpcc_06  = calculate_spatial_rmse(var_06_summer_winter_mag, var_gpcc_summer_winter_mag)
     rmse_gpcc_05  = calculate_spatial_rmse(var_05_summer_winter_mag, var_gpcc_summer_winter_mag)


# Print results for verification
     print("IMERG Correlations:")
     print(f"ICON-08: r = {corr_imerg_08:.3f}, p = {p_value_imerg_08:.2e}")
     print(f"ICON-06: r = {corr_imerg_06:.3f}, p = {p_value_imerg_06:.2e}")
     print(f"ICON-05: r = {corr_imerg_05:.3f}, p = {p_value_imerg_05:.2e}")

     print("\nMSWEP Correlations:")
     print(f"ICON-08: r = {corr_mswep_08:.3f}, p = {p_value_mswep_08:.2e}")
     print(f"ICON-06: r = {corr_mswep_06:.3f}, p = {p_value_mswep_06:.2e}")
     print(f"ICON-05: r = {corr_mswep_05:.3f}, p = {p_value_mswep_05:.2e}")

     print("\nERA5 Correlations:")
     print(f"ICON-08: r = {corr_era5_08:.3f}, p = {p_value_era5_08:.2e}")
     print(f"ICON-06: r = {corr_era5_06:.3f}, p = {p_value_era5_06:.2e}")
     print(f"ICON-05: r = {corr_era5_05:.3f}, p = {p_value_era5_05:.2e}")

     print("\nCMORPH Correlations:")
     print(f"ICON-08: r = {corr_cmorph_08:.3f}, p = {p_value_cmorph_08:.2e}")
     print(f"ICON-06: r = {corr_cmorph_06:.3f}, p = {p_value_cmorph_06:.2e}")
     print(f"ICON-05: r = {corr_cmorph_05:.3f}, p = {p_value_cmorph_05:.2e}")

     print("\nCPC Correlations:")
     print(f"ICON-08: r = {corr_cpc_08:.3f}, p = {p_value_cpc_08:.2e}")
     print(f"ICON-06: r = {corr_cpc_06:.3f}, p = {p_value_cpc_06:.2e}")
     print(f"ICON-05: r = {corr_cpc_05:.3f}, p = {p_value_cpc_05:.2e}")

     print("\nGPCC Correlations:")
     print(f"ICON-08: r = {corr_gpcc_08:.3f}, p = {p_value_gpcc_08:.2e}")
     print(f"ICON-06: r = {corr_gpcc_06:.3f}, p = {p_value_gpcc_06:.2e}")
     print(f"ICON-05: r = {corr_gpcc_05:.3f}, p = {p_value_gpcc_05:.2e}")


     print("IMERG RMSE:")
     print(f"ICON-08: rmse = {rmse_imerg_08:.3f}")
     print(f"ICON-06: rmse = {rmse_imerg_06:.3f}")
     print(f"ICON-05: rmse = {rmse_imerg_05:.3f}")

     
     print("MSWEP RMSE:")
     print(f"ICON-08: rmse = {rmse_mswep_08:.3f}")
     print(f"ICON-06: rmse = {rmse_mswep_06:.3f}")
     print(f"ICON-05: rmse = {rmse_mswep_05:.3f}")


     print("ERA5 RMSE:")
     print(f"ICON-08: rmse = {rmse_era5_08:.3f}")
     print(f"ICON-06: rmse = {rmse_era5_06:.3f}")
     print(f"ICON-05: rmse = {rmse_era5_05:.3f}")

     print("CMORPH RMSE:")
     print(f"ICON-08: rmse = {rmse_cmorph_08:.3f}")
     print(f"ICON-06: rmse = {rmse_cmorph_06:.3f}")
     print(f"ICON-05: rmse = {rmse_cmorph_05:.3f}")


     print("CPC RMSE:")
     print(f"ICON-08: rmse = {rmse_cpc_08:.3f}")
     print(f"ICON-06: rmse = {rmse_cpc_06:.3f}")
     print(f"ICON-05: rmse = {rmse_cpc_05:.3f}")


     print("GPCC RMSE:")
     print(f"ICON-08: rmse = {rmse_gpcc_08:.3f}")
     print(f"ICON-06: rmse = {rmse_gpcc_06:.3f}")
     print(f"ICON-05: rmse = {rmse_gpcc_05:.3f}")

#  Preparing for the figures & Spceifying the Projections
     proj = ccrs.Mercator()
     fig = plt.figure(layout='constrained',figsize=(12, 6))
     # set the spacing between subplots
#     plt.subplots_adjust(left=0.1,
#                    bottom=0.1,
#                    right=0.9,
#                    top=0.1,
#                    wspace=0.4,
#                    hspace=0.1)
#     fig.tight_layout() 
#  Plotting the figures with a 2X2 Matrix
     # Subplot for the Top Left
   
     ax1 = fig.add_subplot(3,3,1, projection=proj)     
     if season=='NH_Summer':
         plot = var_08_summer_winter_mag.plot.contourf(
                      ax=ax1,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm, add_colorbar=False,add_labels=False,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'}, 
                     )
     # Add coast line 
         plot = var_08_summer_winter_NH.plot.contour(ax=ax1,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
         plot = var_08_summer_winter_SH.plot.contour(ax=ax1,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
         ax1.coastlines(resolution='10m', lw=0.51,color='gray')
         ax1.set_title(f'(a) ICON-10KM',fontsize=8,fontweight='bold')
        # ax1.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
         ax1.set_aspect(2)
         ax1.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
         ax1.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present
     gls = ax1.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color="none",linewidth=2,alpha=0.5, linestyle='--')
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}





#     _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)

#   Figure 2
     ax2 = fig.add_subplot(3,3,2, projection=proj)
     if season=='NH_Summer':
         plot = var_06_summer_winter_mag.plot.contourf(
                      ax=ax2,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm, add_colorbar=False,add_labels=False,
 #                     vmin=55,
 #                     vmax=100,
 #                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
 #                                  'extend': 'both',
 #                                  'shrink': .6,
 #                                  'orientation': 'horizontal'},
                     )
     # Add coast line
         plot = var_06_summer_winter_NH.plot.contour(ax=ax2,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
         plot = var_06_summer_winter_SH.plot.contour(ax=ax2,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
         ax2.coastlines(resolution='10m', lw=0.51,color='gray')
         ax2.set_title(f'(b) ICON-40KM',fontsize=8,fontweight='bold')
     #    ax2.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
         ax2.set_aspect(2)
         ax2.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
         ax2.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present
     gls = ax2.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}



#    Figure 3
     ax3 = fig.add_subplot(3,3,3, projection=proj)
     if season=='NH_Summer':
        plot_cf = var_05_summer_winter_mag.plot.contourf(
                      ax=ax3,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'},
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'fraction': 0.046,  # width of the colorbar relative to subplot
                                   'pad': 0.04,        # distance from the subplot
                                   'orientation': 'vertical'},    


                 )
     # Add coast line
        plot = var_05_summer_winter_NH.plot.contour(ax=ax3,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_05_summer_winter_SH.plot.contour(ax=ax3,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax3.coastlines(resolution='10m', lw=0.51,color='gray')
        ax3.set_title(f'(c) ICON-80KM',fontsize=8,fontweight='bold')
    #    ax3.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax3.set_aspect(2)
        ax3.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax3.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present
        plot_cf.colorbar.ax.tick_params(labelsize=10)
        plot_cf.colorbar.ax.set_ylabel('Total Precipitation (mm/day)', fontsize=8) 
    

     gls = ax3.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}



# Figure 4 

     ax4 = fig.add_subplot(3,3,4, projection=proj)
     if season=='NH_Summer':
        plot = var_era5_summer_winter_mag.plot.contourf(
                      ax=ax4,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm, add_colorbar=False,add_labels=False,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        plot = var_era5_summer_winter_NH.plot.contour(ax=ax4,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_era5_summer_winter_SH.plot.contour(ax=ax4,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax4.coastlines(resolution='10m', lw=0.51,color='gray')
        ax4.set_title(f'(d) ERA5 (Observation)',fontsize=8,fontweight='bold')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax4.set_aspect(2)
        ax4.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax4.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present
     gls = ax4.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}
  
 
     # Figure 5

     ax5 = fig.add_subplot(3,3,5, projection=proj)
     if season=='NH_Summer':
        plot = var_imerg_summer_winter_mag.plot.contourf(
                      ax=ax5,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm, add_colorbar=False,add_labels=False,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        plot = var_imerg_summer_winter_NH.plot.contour(ax=ax5,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_imerg_summer_winter_SH.plot.contour(ax=ax5,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax5.coastlines(resolution='10m', lw=0.51,color='gray')
        ax5.set_title(f'(e) IMERG (Observation)',fontsize=8,fontweight='bold')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax5.set_aspect(2)
        ax5.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax5.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present

     gls = ax5.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}



     # Figure 6

     ax6 = fig.add_subplot(3,3,6, projection=proj)
     if season=='NH_Summer':
        plot_cf = var_cpc_summer_winter_mag.plot.contourf(
                      ax=ax6,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'},
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'fraction': 0.046,  # width of the colorbar relative to subplot
                                   'pad': 0.04,        # distance from the subplot
                                   'orientation': 'vertical'},



                     )
     # Add coast line
        plot = var_cpc_summer_winter_NH.plot.contour(ax=ax6,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_cpc_summer_winter_SH.plot.contour(ax=ax6,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax6.coastlines(resolution='10m', lw=0.51,color='gray')
        ax6.set_title(f'(f) CPC (Observation)',fontsize=8,fontweight='bold')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax6.set_aspect(2)
        ax6.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax6.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present
        plot_cf.colorbar.ax.tick_params(labelsize=10)
        plot_cf.colorbar.ax.set_ylabel('Total Precipitation (mm/day)', fontsize=8)

     gls = ax6.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}



     # Figure 7

     ax7 = fig.add_subplot(3,3,7, projection=proj)
     if season=='NH_Summer':
        plot = var_gpcc_summer_winter_mag.plot.contourf(
                      ax=ax7,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm,add_colorbar=False,add_labels=False,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        plot = var_gpcc_summer_winter_NH.plot.contour(ax=ax7,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_gpcc_summer_winter_SH.plot.contour(ax=ax7,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax7.coastlines(resolution='10m', lw=0.51,color='gray')
        ax7.set_title(f'(g) GPCC (Observation)',fontsize=8,fontweight='bold')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax7.set_aspect(2)
        ax7.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax7.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present


     gls = ax7.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}


 # Figure 8

     ax8 = fig.add_subplot(3,3,8, projection=proj)
     if season=='NH_Summer':
        plot = var_mswep_summer_winter_mag.plot.contourf(
                          ax=ax8,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm,add_colorbar=False,add_labels=False,
#                      vmin=55,
#                      vmax=100,
#                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
#                                   'extend': 'both',
#                                   'shrink': .6,
#                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        plot = var_mswep_summer_winter_NH.plot.contour(ax=ax8,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_mswep_summer_winter_SH.plot.contour(ax=ax8,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax8.coastlines(resolution='10m', lw=0.51,color='gray')
        ax8.set_title(f'(h) MSWEP (Observation)',fontsize=8,fontweight='bold')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax8.set_aspect(2)
        ax8.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax8.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present


     gls = ax8.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}



     ax9 = fig.add_subplot(3,3,9, projection=proj)
     if season=='NH_Summer':
        plot_cf = var_cmorph_summer_winter_mag.plot.contourf(
                      ax=ax9,
                      transform=ccrs.PlateCarree(),
                      cmap='Spectral_r',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'fraction': 0.046,  # width of the colorbar relative to subplot
    				   'pad': 0.04,        # distance from the subplot
                                   'orientation': 'vertical'},
                     )
     # Add coast line
        plot = var_cmorph_summer_winter_NH.plot.contour(ax=ax9,levels=[2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        plot = var_cmorph_summer_winter_SH.plot.contour(ax=ax9,levels=[-2.0], transform=ccrs.PlateCarree(), colors=['k'],linewidths=0.2 )
        ax9.coastlines(resolution='10m', lw=0.51,color='gray')
        ax9.set_title(f'(i) CMORP (Observation)',fontsize=8,fontweight='bold')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax9.set_aspect(2)
        plot_cf.colorbar.ax.tick_params(labelsize=10)
        plot_cf.colorbar.ax.set_ylabel('Total Precipitation (mm/day)', fontsize=8)
  
        ax9.tick_params(axis='both', which='major', labelsize=6)  # reduce major tick labels
        ax9.tick_params(axis='both', which='minor', labelsize=6)   # reduce minor tick labels if present

     gls = ax9.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False
     gls.xlabel_style = {'size': 6, 'weight': 'bold'}
     gls.ylabel_style = {'size': 6, 'weight': 'bold'}


#Save the Figure in png format
     plt.savefig(plot_name,dpi=600, bbox_inches='tight', pad_inches=0.1)
     fig.savefig("absolute_GM_precipitation.pdf", bbox_inches='tight', pad_inches=0.1)


     ds_to_save = xr.Dataset()

     # Add all your calculated variables to the dataset
     ds_to_save['var_08_summer_winter_mag'] = var_08_summer_winter_mag
     ds_to_save['var_08_summer_winter_NH'] = var_08_summer_winter_NH
     ds_to_save['var_08_summer_winter_SH'] = var_08_summer_winter_SH

     ds_to_save['var_06_summer_winter_mag'] = var_06_summer_winter_mag
     ds_to_save['var_06_summer_winter_NH'] = var_06_summer_winter_NH
     ds_to_save['var_06_summer_winter_SH'] = var_06_summer_winter_SH

     ds_to_save['var_05_summer_winter_mag'] = var_05_summer_winter_mag
     ds_to_save['var_05_summer_winter_NH'] = var_05_summer_winter_NH
     ds_to_save['var_05_summer_winter_SH'] = var_05_summer_winter_SH

     ds_to_save['var_imerg_summer_winter_mag'] = var_imerg_summer_winter_mag
     ds_to_save['var_imerg_summer_winter_NH'] = var_imerg_summer_winter_NH
     ds_to_save['var_imerg_summer_winter_SH'] = var_imerg_summer_winter_SH


###  Add subplot names in python


# ---------------------------------------------------------------------------------
# 4. Add attributes for clarity
# ---------------------------------------------------------------------------------
     ds_to_save.attrs['description'] = 'Calculated summer-winter magnitude and NH/SH components for ICON models and IMERG'
     ds_to_save.attrs['created'] = str(np.datetime64('now'))

# ---------------------------------------------------------------------------------
# 5. Save to a NetCDF file
# ---------------------------------------------------------------------------------
     output_filename = 'calculated_data_for_plotting.nc'
     ds_to_save.to_netcdf(output_filename)
     print(f"All data successfully saved to {output_filename}")
