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

from scipy.signal import butter, filtfilt


##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

#Here choose NH_Summer or NH_Winter or NH_Summer-NH_Winter
season='NH_Summer'

Region="Indian_Core"

plot_name="Indian_Core"

plot_var_key="tot_prec"

fig, ax = plt.subplots(2, 2, figsize=(12, 12),layout='constrained')

reg = 'South_Asia','Africa','America','Australia'



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


if Region == 'Indian_Core':
    # Select a Region
    #South Asia
    lat_min=18
    lat_max=25
    lon_min=72
    lon_max=85
    plot_name='Indian_Core'

elif Region == 'North_Africa':
    lat_min=8
    lat_max=14
    lon_min=-38
    lon_max=40
    plot_name='North_Africa'

elif Region == 'South_Africa':
    lat_min=-22
    lat_max=-10
    lon_min=20
    lon_max=40
    plot_name='South_Africa'

elif Region == 'North_America':
    lat_min=10
    lat_max=20
    lon_min=-118
    lon_max=-90
    plot_name='North_America'

elif Region == 'South_America':
    lat_min=-20
    lat_max=-10
    lon_min=-75
    lon_max=-40
    plot_name='South_America'

elif Region == 'Australia':
    lat_min=-20
    lat_max=-15
    lon_min=120
    lon_max=145
    plot_name='Australia'


# base directory where your analysis data is stored GPU
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

data_base_dir_imerg = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/IMERG/day_nc4_files/post_processed_files'
model_data_imerg    =  'precipitation_cdo_all_rmp_10years.nc'



# netCDF variable name of variable to plot
var='tot_prec','t'

# Read the Model Data from Here
model_data_dir_08    = os.path.join(data_base_dir_08, model_data_08)
model_data_dir_06    = os.path.join(data_base_dir_06, model_data_06)
model_data_dir_05    = os.path.join(data_base_dir_05, model_data_05)
model_data_dir_imerg    = os.path.join(data_base_dir_imerg, model_data_imerg)


ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_imerg  = xr.open_dataset(model_data_dir_imerg)

# Read the Variable from Here.
var_08 = ds_08[plot_var_key]
var_06 = ds_06[plot_var_key]
var_05 = ds_05[plot_var_key]
var_imerg = ds_imerg['precipitation']
var_imerg = var_imerg.transpose()

### Crop the Lat-Lon Section for Tropical Areas.  

var_08 = var_08.sel(lat=slice(-40,40))
var_06 = var_06.sel(lat=slice(-40,40))
var_05 = var_05.sel(lat=slice(-40,40))
var_imerg = var_imerg.sel(lat=slice(-40,40))

## 
# Mask Data Sets corresponding to 55% min contribution of Summer Season to Annual Contribution
# Mask Data Sets corresponding to 55% min contribution of Summer Season to Annual Contribution
data_base_mask = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig1_domains/original_masks'
mask_08='08_mask_2mm_55.nc'
mask_06='06_mask_2mm_55.nc'
mask_05='05_mask_2mm_55.nc'
mask_imerg='imerg_mask_2mm_55.nc'

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_imerg = os.path.join(data_base_mask,mask_imerg)


ds_mask_08 = xr.open_dataset(data_dir_mask_08)
ds_mask_06 = xr.open_dataset(data_dir_mask_06)
ds_mask_05 = xr.open_dataset(data_dir_mask_05)
ds_mask_imerg = xr.open_dataset(data_dir_mask_imerg)

mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
mask_imerg = ds_mask_imerg['__xarray_dataarray_variable__']


# Mutiply the Mask with the actual values. 

var_08 = var_08*mask_08
var_06 = var_06*mask_06
var_05 = var_05*mask_05
var_imerg = var_imerg*mask_imerg


## Transformed

var_08_transformed = np.log1p(var_08)  # or np.sqrt(var_08)
var_06_transformed = np.log1p(var_06)
var_05_transformed = np.log1p(var_05)
var_imerg_transformed = np.log1p(var_imerg)


# We calculate Only Norther Hemisphere Interannual Variability

var_08_summer = var_08_transformed.sel(time=var_08_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_08_absolute = var_08.sel(time=var_08_transformed.time.dt.month.isin([6, 7, 8, 9]))

var_06_summer = var_06_transformed.sel(time=var_06_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_06_absolute = var_06.sel(time=var_06_transformed.time.dt.month.isin([6, 7, 8, 9]))

var_05_summer = var_05_transformed.sel(time=var_05_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_05_absolute = var_05.sel(time=var_05_transformed.time.dt.month.isin([6, 7, 8, 9]))

var_imerg_summer = var_imerg_transformed.sel(time=var_imerg_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_imerg_absolute = var_imerg.sel(time=var_imerg_transformed.time.dt.month.isin([6, 7, 8, 9]))

###

# India

lat_min_core=18
lat_max_core=25
lon_min_core=72
lon_max_core=85

# Africa. 

#lat_min_core=10
#lat_max_core=20
#lon_min_core=-18
#lon_max_core=16

###  

var_08_core = var_08_absolute.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))
var_06_core = var_06_absolute.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))
var_05_core = var_05_absolute.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))
var_obs_imerg_core = var_imerg_absolute.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))

###

var_08_transformed_core = var_08_summer.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))
var_06_transformed_core = var_06_summer.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))
var_05_transformed_core = var_05_summer.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))
var_obs_imerg_transformed_core = var_imerg_summer.sel(lat=slice(lat_min_core,lat_max_core),lon=slice(lon_min_core,lon_max_core))


##

import numpy as np

# Take ICON 10km absolute and transformed
raw_data = var_obs_imerg_core.values.flatten()      # Absolute precipitation
trans_data = var_obs_imerg_transformed_core.values.flatten()  # Log-transformed

# Remove NaN values
mask = ~np.isnan(raw_data) & ~np.isnan(trans_data)
raw_data = raw_data[mask]
trans_data = trans_data[mask]

# Calculate key percentiles
percentiles = [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
raw_perc = np.percentile(raw_data, percentiles)
trans_perc = np.percentile(trans_data, percentiles)

# Simple print of what happens
print("ICON 10km - Raw vs Transformed Precipitation")
print("=" * 60)
print(f"{'Percentile':<10} {'Raw (mm/day)':<12} {'Transformed':<12}")
print("-" * 60)

for p, raw_val, trans_val in zip(percentiles, raw_perc, trans_perc):
    print(f"{p}%:       {raw_val:>8.2f}     {trans_val:>8.2f}")

# Calculate compression at heavy rain threshold
print("\n" + "=" * 60)
print("COMPRESSION ANALYSIS")
print("=" * 60)

# How much is a heavy rain event compressed?
heavy_threshold = 20  # mm/day - adjust based on your data

# Find closest raw value to threshold
idx = np.argmin(np.abs(raw_data - heavy_threshold))
raw_at_threshold = raw_data[idx]
trans_at_threshold = trans_data[idx]
median_raw = raw_perc[2]  # 50th percentile
median_trans = trans_perc[2]

# Compression calculation
raw_diff = raw_at_threshold - median_raw
trans_diff = trans_at_threshold - median_trans
compression = raw_diff / trans_diff if trans_diff != 0 else np.inf

print(f"Heavy rain threshold: {heavy_threshold} mm/day")
print(f"Closest actual value: {raw_at_threshold:.1f} mm/day")
print(f"Transformed value: {trans_at_threshold:.2f}")
print(f"Compression factor: {compression:.1f}x")
print(f"→ A {raw_at_threshold:.0f} mm/day event has {compression:.1f}x")
print(f"  less influence on variance after transformation")

# Quick stats
print("\n" + "=" * 60)
print("QUICK STATS")
print("=" * 60)
print(f"Total data points: {len(raw_data):,}")
print(f"Dry days (0 mm): {np.sum(raw_data == 0):,} ({100*np.mean(raw_data == 0):.1f}%)")
print(f"Days > 10mm: {np.sum(raw_data > 10):,} ({100*np.mean(raw_data > 10):.1f}%)")
print(f"Days > 50mm: {np.sum(raw_data > 50):,} ({100*np.mean(raw_data > 50):.1f}%)")
