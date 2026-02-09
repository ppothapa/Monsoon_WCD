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


# Observational Data Sets for CPC

data_base_dir_cpc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CPC_data/daily'
model_data_cpc    =  'precip.daily_rmp_10years.nc'

# Observational Data Sets for ERA5

data_base_dir_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/daily'
model_data_era5    =  'pr_day_reanalysis_era5_r1i1p1_daily_rmp_10years.nc'


# Obesrvational Data Set for MSWEP

data_base_dir_mswep = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/MSWEP/Data_2006_2022'
model_data_mswep    = 'precip_rmp_10years.nc'

# CMORPH Data ((Time step = 3653))

data_base_dir_cmorph = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CMORPH/data/CMORPH/daily_nc'
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
var_imerg = var_imerg.transpose()

### Crop the Lat-Lon Section for Tropical Areas.  

var_08 = var_08.sel(lat=slice(-40,40))
var_06 = var_06.sel(lat=slice(-40,40))
var_05 = var_05.sel(lat=slice(-40,40))
var_imerg = var_imerg.sel(lat=slice(-40,40))
var_cpc = var_cpc.sel(lat=slice(-40,40))
var_era5 = var_era5.sel(lat=slice(-40,40))
var_mswep = var_mswep.sel(lat=slice(-40,40))
var_cmorph = var_cmorph.sel(lat=slice(-40,40))
var_gpcc = var_gpcc.sel(lat=slice(-40,40))

## 
# Mask Data Sets corresponding to 55% min contribution of Summer Season to Annual Contribution
# Mask Data Sets corresponding to 55% min contribution of Summer Season to Annual Contribution
data_base_mask = '/capstor/store/cscs/userlab/cwp03/ppothapa/Paper_1_10km_Monsoon/Fig1_domains/original_masks'
mask_08='08_mask_2mm_55.nc'
mask_06='06_mask_2mm_55.nc'
mask_05='05_mask_2mm_55.nc'
mask_era5='era5_mask_2mm_55.nc'
mask_cpc='cpc_mask_2mm_55.nc'
mask_imerg='imerg_mask_2mm_55.nc'
mask_mswep='mswep_mask_2mm_55.nc'
mask_cmorph='cmorph_mask_2mm_55.nc'
mask_gpcc='gpcc_mask_2mm_55.nc'

data_dir_mask_08 = os.path.join(data_base_mask,mask_08)
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_imerg = os.path.join(data_base_mask,mask_imerg)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)
data_dir_mask_cmorph = os.path.join(data_base_mask,mask_cmorph)
data_dir_mask_gpcc = os.path.join(data_base_mask,mask_gpcc)


ds_mask_08 = xr.open_dataset(data_dir_mask_08)
ds_mask_06 = xr.open_dataset(data_dir_mask_06)
ds_mask_05 = xr.open_dataset(data_dir_mask_05)
ds_mask_era5 = xr.open_dataset(data_dir_mask_era5)
ds_mask_imerg = xr.open_dataset(data_dir_mask_imerg)
ds_mask_cpc = xr.open_dataset(data_dir_mask_cpc)
ds_mask_mswep = xr.open_dataset(data_dir_mask_mswep)
ds_mask_cmorph = xr.open_dataset(data_dir_mask_cmorph)
ds_mask_gpcc = xr.open_dataset(data_dir_mask_gpcc)

mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
mask_era5 = ds_mask_era5['__xarray_dataarray_variable__']
mask_imerg = ds_mask_imerg['__xarray_dataarray_variable__']
mask_cpc = ds_mask_cpc['__xarray_dataarray_variable__']
mask_mswep = ds_mask_mswep['__xarray_dataarray_variable__']
mask_cmorph = ds_mask_cmorph['__xarray_dataarray_variable__']
mask_gpcc = ds_mask_gpcc['__xarray_dataarray_variable__']


# Mutiply the Mask with the actual values. 

var_08 = var_08*mask_08
var_06 = var_06*mask_06
var_05 = var_05*mask_05
var_imerg = var_imerg*mask_imerg
var_cpc = var_cpc*mask_cpc
var_era5 = var_era5*mask_era5
var_mswep = var_mswep*mask_mswep
var_cmorph = var_cmorph*mask_cmorph
var_gpcc = var_gpcc*mask_gpcc


## Transformed

var_08_transformed = np.log1p(var_08)  # or np.sqrt(var_08)
var_06_transformed = np.log1p(var_06)
var_05_transformed = np.log1p(var_05)
var_imerg_transformed = np.log1p(var_imerg)
var_era5_transformed = np.log1p(var_era5)
var_cpc_transformed = np.log1p(var_cpc)
var_mswep_transformed = np.log1p(var_mswep)
var_cmorph_transformed = np.log1p(var_cmorph)
var_gpcc_transformed = np.log1p(var_gpcc)



# We calculate Only Norther Hemisphere Interannual Variability

var_08_summer = var_08_transformed.sel(time=var_08_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_08_winter = var_08_transformed.sel(time=var_08_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_06_summer = var_06_transformed.sel(time=var_06_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_06_winter = var_06_transformed.sel(time=var_06_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_05_summer = var_05_transformed.sel(time=var_05_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_05_winter = var_05_transformed.sel(time=var_05_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_imerg_summer = var_imerg_transformed.sel(time=var_imerg_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_imerg_winter = var_imerg_transformed.sel(time=var_imerg_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_era5_summer = var_era5_transformed.sel(time=var_era5_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_era5_winter = var_era5_transformed.sel(time=var_era5_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_cpc_summer = var_cpc_transformed.sel(time=var_cpc_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_cpc_winter = var_cpc_transformed.sel(time=var_cpc_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_mswep_summer = var_mswep_transformed.sel(time=var_mswep_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_mswep_winter = var_mswep_transformed.sel(time=var_mswep_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_cmorph_summer = var_cmorph_transformed.sel(time=var_cmorph_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_cmorph_winter = var_cmorph_transformed.sel(time=var_cmorph_transformed.time.dt.month.isin([12, 1, 2, 3]))

var_gpcc_summer = var_gpcc_transformed.sel(time=var_gpcc_transformed.time.dt.month.isin([6, 7, 8, 9]))
var_gpcc_winter = var_gpcc_transformed.sel(time=var_gpcc_transformed.time.dt.month.isin([12, 1, 2, 3]))


#Resample this by Summer Years

evolve_var_08_summer_intraseasonal  = apply_bandpass(var_08_summer)
evolve_var_08_winter_intraseasonal  = apply_bandpass(var_08_winter)
print(evolve_var_08_summer_intraseasonal.shape)

evolve_var_06_summer_intraseasonal  = apply_bandpass(var_06_summer)
evolve_var_06_winter_intraseasonal  = apply_bandpass(var_06_winter)
print(evolve_var_06_summer_intraseasonal.shape)

evolve_var_05_summer_intraseasonal  = apply_bandpass(var_05_summer)
evolve_var_05_winter_intraseasonal  = apply_bandpass(var_05_winter)
print(evolve_var_05_summer_intraseasonal.shape)

evolve_var_era5_summer_intraseasonal = apply_bandpass(var_era5_summer * 86400)
evolve_var_era5_winter_intraseasonal = apply_bandpass(var_era5_winter * 86400)
print(evolve_var_era5_summer_intraseasonal.shape)

evolve_var_imerg_summer_intraseasonal = apply_bandpass(var_imerg_summer)
evolve_var_imerg_winter_intraseasonal = apply_bandpass(var_imerg_winter)
print(evolve_var_imerg_summer_intraseasonal.shape)

evolve_var_cpc_summer_intraseasonal = apply_bandpass(var_cpc_summer)
evolve_var_cpc_winter_intraseasonal = apply_bandpass(var_cpc_winter)
print(evolve_var_cpc_summer_intraseasonal.shape)

evolve_var_mswep_summer_intraseasonal = apply_bandpass(var_mswep_summer)
evolve_var_mswep_winter_intraseasonal = apply_bandpass(var_mswep_winter)
print(evolve_var_mswep_summer_intraseasonal.shape)

evolve_var_cmorph_summer_intraseasonal = apply_bandpass(var_cmorph_summer)
evolve_var_cmorph_winter_intraseasonal = apply_bandpass(var_cmorph_winter)
print(evolve_var_cmorph_summer_intraseasonal.shape)

evolve_var_gpcc_summer_intraseasonal = apply_bandpass(var_gpcc_summer)
evolve_var_gpcc_winter_intraseasonal = apply_bandpass(var_gpcc_winter)
print(evolve_var_gpcc_summer_intraseasonal.shape)


## Save Intraseasonal Variability Files as Well in the RawFormat. 


evolve_var_08_summer_intraseasonal.to_netcdf("evolve_var_08_summer_intraseasonal.nc")
evolve_var_06_summer_intraseasonal.to_netcdf("evolve_var_06_summer_intraseasonal.nc")
evolve_var_05_summer_intraseasonal.to_netcdf("evolve_var_05_summer_intraseasonal.nc")
evolve_var_imerg_summer_intraseasonal.to_netcdf("evolve_var_imerg_summer_intraseasonal.nc")
evolve_var_era5_summer_intraseasonal.to_netcdf("evolve_var_era5_summer_intraseasonal.nc")
evolve_var_cpc_summer_intraseasonal.to_netcdf("evolve_var_cpc_summer_intraseasonal.nc")
evolve_var_mswep_summer_intraseasonal.to_netcdf("evolve_var_mswep_summer_intraseasonal.nc")
evolve_var_cmorph_summer_intraseasonal.to_netcdf("evolve_var_cmorph_summer_intraseasonal.nc")
evolve_var_gpcc_summer_intraseasonal.to_netcdf("evolve_var_gpcc_summer_intraseasonal.nc")


##### Save the Intraseasoanl Files As well.  

evolve_var_08_winter_intraseasonal.to_netcdf("evolve_var_08_winter_intraseasonal.nc")
evolve_var_06_winter_intraseasonal.to_netcdf("evolve_var_06_winter_intraseasonal.nc")
evolve_var_05_winter_intraseasonal.to_netcdf("evolve_var_05_winter_intraseasonal.nc")
evolve_var_imerg_winter_intraseasonal.to_netcdf("evolve_var_imerg_winter_intraseasonal.nc")
evolve_var_era5_winter_intraseasonal.to_netcdf("evolve_var_era5_winter_intraseasonal.nc")
evolve_var_cpc_winter_intraseasonal.to_netcdf("evolve_var_cpc_winter_intraseasonal.nc")
evolve_var_mswep_winter_intraseasonal.to_netcdf("evolve_var_mswep_winter_intraseasonal.nc")
evolve_var_cmorph_winter_intraseasonal.to_netcdf("evolve_var_cmorph_winter_intraseasonal.nc")
evolve_var_gpcc_winter_intraseasonal.to_netcdf("evolve_var_gpcc_winter_intraseasonal.nc")



######


# This is the Climatological Mean of the Summer/Winter Precipitatio
evolve_var_08_summer_intraseasonal_std = evolve_var_08_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_06_summer_intraseasonal_std = evolve_var_06_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_05_summer_intraseasonal_std = evolve_var_05_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_imerg_summer_intraseasonal_std = evolve_var_imerg_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_era5_summer_intraseasonal_std = evolve_var_era5_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_cpc_summer_intraseasonal_std = evolve_var_cpc_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_mswep_summer_intraseasonal_std = evolve_var_mswep_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_cmorph_summer_intraseasonal_std = evolve_var_cmorph_summer_intraseasonal.std(dim='time', skipna=True)
evolve_var_gpcc_summer_intraseasonal_std = evolve_var_gpcc_summer_intraseasonal.std(dim='time', skipna=True)

evolve_var_08_winter_intraseasonal_std = evolve_var_08_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_06_winter_intraseasonal_std = evolve_var_06_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_05_winter_intraseasonal_std = evolve_var_05_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_imerg_winter_intraseasonal_std = evolve_var_imerg_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_era5_winter_intraseasonal_std = evolve_var_era5_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_cpc_winter_intraseasonal_std = evolve_var_cpc_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_mswep_winter_intraseasonal_std = evolve_var_mswep_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_cmorph_winter_intraseasonal_std = evolve_var_cmorph_winter_intraseasonal.std(dim='time', skipna=True)
evolve_var_gpcc_winter_intraseasonal_std = evolve_var_gpcc_winter_intraseasonal.std(dim='time', skipna=True)

#  Save the netcdf data for standard deviation. 

evolve_var_08_summer_intraseasonal_std.to_netcdf("evolve_var_08_summer_intraseasonal_std.nc")
evolve_var_06_summer_intraseasonal_std.to_netcdf("evolve_var_06_summer_intraseasonal_std.nc")
evolve_var_05_summer_intraseasonal_std.to_netcdf("evolve_var_05_summer_intraseasonal_std.nc")
evolve_var_imerg_summer_intraseasonal_std.to_netcdf("evolve_var_imerg_summer_intraseasonal_std.nc")
evolve_var_era5_summer_intraseasonal_std.to_netcdf("evolve_var_era5_summer_intraseasonal_std.nc")
evolve_var_cpc_summer_intraseasonal_std.to_netcdf("evolve_var_cpc_summer_intraseasonal_std.nc")
evolve_var_mswep_summer_intraseasonal_std.to_netcdf("evolve_var_mswep_summer_intraseasonal_std.nc")
evolve_var_cmorph_summer_intraseasonal_std.to_netcdf("evolve_var_cmorph_summer_intraseasonal_std.nc")
evolve_var_gpcc_summer_intraseasonal_std.to_netcdf("evolve_var_gpcc_summer_intraseasonal_std.nc")


#   Here calculate the Interannal Variability Variance

# This is the Climatological Mean of the Summer/Winter Precipitatio
evolve_var_08_summer_intraseasonal_var = evolve_var_08_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_06_summer_intraseasonal_var = evolve_var_06_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_05_summer_intraseasonal_var = evolve_var_05_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_imerg_summer_intraseasonal_var = evolve_var_imerg_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_era5_summer_intraseasonal_var = evolve_var_era5_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_cpc_summer_intraseasonal_var = evolve_var_cpc_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_mswep_summer_intraseasonal_var = evolve_var_mswep_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_cmorph_summer_intraseasonal_var = evolve_var_cmorph_summer_intraseasonal.var(dim='time', skipna=True)
evolve_var_gpcc_summer_intraseasonal_var = evolve_var_gpcc_summer_intraseasonal.var(dim='time', skipna=True)

evolve_var_08_winter_intraseasonal_var = evolve_var_08_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_06_winter_intraseasonal_var = evolve_var_06_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_05_winter_intraseasonal_var = evolve_var_05_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_imerg_winter_intraseasonal_var = evolve_var_imerg_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_era5_winter_intraseasonal_var = evolve_var_era5_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_cpc_winter_intraseasonal_var = evolve_var_cpc_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_mswep_winter_intraseasonal_var = evolve_var_mswep_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_cmorph_winter_intraseasonal_var = evolve_var_cmorph_winter_intraseasonal.var(dim='time', skipna=True)
evolve_var_gpcc_winter_intraseasonal_var = evolve_var_gpcc_winter_intraseasonal.var(dim='time', skipna=True)
### Saving the Data

evolve_var_08_summer_intraseasonal_var.to_netcdf("evolve_var_08_summer_intraseasonal_var.nc")
evolve_var_06_summer_intraseasonal_var.to_netcdf("evolve_var_06_summer_intraseasonal_var.nc")
evolve_var_05_summer_intraseasonal_var.to_netcdf("evolve_var_05_summer_intraseasonal_var.nc")
evolve_var_imerg_summer_intraseasonal_var.to_netcdf("evolve_var_imerg_summer_intraseasonal_var.nc")
evolve_var_era5_summer_intraseasonal_var.to_netcdf("evolve_var_era5_summer_intraseasonal_var.nc")
evolve_var_cpc_summer_intraseasonal_var.to_netcdf("evolve_var_cpc_summer_intraseasonal_var.nc")
evolve_var_mswep_summer_intraseasonal_var.to_netcdf("evolve_var_mswep_summer_intraseasonal_var.nc")
evolve_var_cmorph_summer_intraseasonal_var.to_netcdf("evolve_var_cmorph_summer_intraseasonal_var.nc")
evolve_var_gpcc_summer_intraseasonal_var.to_netcdf("evolve_var_gpcc_summer_intraseasonal_var.nc")

