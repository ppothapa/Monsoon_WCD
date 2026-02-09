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
##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################
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



data_dir_mask_08 = os.path.join(data_base_mask,mask_08)  
data_dir_mask_06 = os.path.join(data_base_mask,mask_06)
data_dir_mask_05 = os.path.join(data_base_mask,mask_05)
data_dir_mask_era5 = os.path.join(data_base_mask,mask_era5)
data_dir_mask_cpc = os.path.join(data_base_mask,mask_cpc)
data_dir_mask_imerg = os.path.join(data_base_mask,mask_imerg)
data_dir_mask_mswep = os.path.join(data_base_mask,mask_mswep)
data_dir_mask_cmorph = os.path.join(data_base_mask,mask_cmorph)

#Read the OpenDataset. 
ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_imerg  = xr.open_dataset(model_data_dir_imerg)
ds_cpc  = xr.open_dataset(model_data_dir_cpc)
ds_era5  = xr.open_dataset(model_data_dir_era5)
ds_mswep  = xr.open_dataset(model_data_dir_mswep)
ds_cmorph  = xr.open_dataset(model_data_dir_cmorph)




ds_mask_08 = xr.open_dataset(data_dir_mask_08)
ds_mask_06 = xr.open_dataset(data_dir_mask_06)
ds_mask_05 = xr.open_dataset(data_dir_mask_05)
ds_mask_era5 = xr.open_dataset(data_dir_mask_era5)
ds_mask_imerg = xr.open_dataset(data_dir_mask_imerg)
ds_mask_cpc = xr.open_dataset(data_dir_mask_cpc)
ds_mask_mswep = xr.open_dataset(data_dir_mask_mswep)
ds_mask_cmorph = xr.open_dataset(data_dir_mask_cmorph)


# Read the Variable from Here.  
var_08 = ds_08["tot_prec"]
var_06 = ds_06["tot_prec"]
var_05 = ds_05["tot_prec"] 
var_imerg = ds_imerg['precipitation']
var_cpc = ds_cpc['precip']
var_era5 = ds_era5['pr']
var_mswep = ds_mswep['precipitation']
var_cmorph = ds_cmorph['cmorph']




mask_08 = ds_mask_08["__xarray_dataarray_variable__"]
mask_06 = ds_mask_06["__xarray_dataarray_variable__"]
mask_05 = ds_mask_05["__xarray_dataarray_variable__"]
mask_era5 = ds_mask_era5['__xarray_dataarray_variable__']
mask_imerg = ds_mask_imerg['__xarray_dataarray_variable__']
mask_cpc = ds_mask_cpc['__xarray_dataarray_variable__']
mask_mswep = ds_mask_mswep['__xarray_dataarray_variable__']
mask_cmorph = ds_mask_cmorph['__xarray_dataarray_variable__']



# IMERG DATA 
    
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



# Select the Seasons and Mean over the Seasons.      
var_08_summer = var_08.sel(time=var_08.time.dt.month.isin([6, 7, 8, 9]))
var_08_winter = var_08.sel(time=var_08.time.dt.month.isin([12, 1, 2, 3]))

var_06_summer = var_06.sel(time=var_06.time.dt.month.isin([6, 7, 8, 9]))
var_06_winter = var_06.sel(time=var_06.time.dt.month.isin([12, 1, 2, 3]))

var_05_summer = var_05.sel(time=var_05.time.dt.month.isin([6, 7, 8, 9]))
var_05_winter = var_05.sel(time=var_05.time.dt.month.isin([12, 1, 2, 3]))
    
var_imerg_summer = var_imerg.sel(time=var_imerg.time.dt.month.isin([6, 7, 8, 9]))
var_imerg_winter = var_imerg.sel(time=var_imerg.time.dt.month.isin([12, 1, 2, 3]))

var_cpc_summer = var_cpc.sel(time=var_cpc.time.dt.month.isin([6, 7, 8, 9]))
var_cpc_winter = var_cpc.sel(time=var_cpc.time.dt.month.isin([12, 1, 2, 3]))

var_era5_summer = var_era5.sel(time=var_era5.time.dt.month.isin([6, 7, 8, 9]))
var_era5_winter = var_era5.sel(time=var_era5.time.dt.month.isin([12, 1, 2, 3]))

var_mswep_summer = var_mswep.sel(time=var_mswep.time.dt.month.isin([6, 7, 8, 9]))
var_mswep_winter = var_mswep.sel(time=var_mswep.time.dt.month.isin([12, 1, 2, 3]))


var_cmorph_summer = var_cmorph.sel(time=var_cmorph.time.dt.month.isin([6, 7, 8, 9]))
var_cmorph_winter = var_cmorph.sel(time=var_cmorph.time.dt.month.isin([12, 1, 2, 3]))



# Summer Season

cli_08_summer  = var_08_summer.mean("time")
cli_08_winter  = var_08_winter.mean("time")
    
cli_06_summer  = var_06_summer.mean("time")
cli_06_winter  = var_06_winter.mean("time") 

cli_05_summer  = var_05_summer.mean("time")
cli_05_winter  = var_05_winter.mean("time")

cli_imerg_summer  = var_imerg_summer.mean("time") 
cli_imerg_winter  = var_imerg_winter.mean("time") 

cli_cpc_summer  = var_cpc_summer.mean("time")
cli_cpc_winter  = var_cpc_winter.mean("time")

cli_era5_summer  = var_era5_summer.mean("time") * 86400
cli_era5_winter  = var_era5_winter.mean("time") * 86400

cli_mswep_summer  = var_mswep_summer.mean("time") 
cli_mswep_winter  = var_mswep_winter.mean("time") 

cli_cmorph_summer  = var_cmorph_summer.mean("time")
cli_cmorph_winter  = var_cmorph_winter.mean("time")



#  Save these also in the NETCDF formats
     
cli_08_summer.to_netcdf("cli_08_summer.nc")
cli_06_summer.to_netcdf("cli_06_summer.nc")
cli_05_summer.to_netcdf("cli_05_summer.nc")
cli_imerg_summer.to_netcdf("cli_imerg_summer.nc")
cli_cpc_summer.to_netcdf("cli_cpc_summer.nc")
cli_era5_summer.to_netcdf("cli_era5_summer.nc")
cli_mswep_summer.to_netcdf("cli_mswep_summer.nc")
cli_cmorph_summer.to_netcdf("cli_cmorph_summer.nc")


# Absolute Percentage Differences

diff_sum_08_obs_imerg  = cli_08_summer - cli_imerg_summer 
diff_sum_06_obs_imerg  = cli_06_summer - cli_imerg_summer
diff_sum_05_obs_imerg  = cli_05_summer - cli_imerg_summer

# Relative Differences

diff_sum_08_obs_imerg_rl  = ((diff_sum_08_obs_imerg ) / cli_imerg_summer ) * 100
diff_sum_06_obs_imerg_rl  = ((diff_sum_06_obs_imerg ) / cli_imerg_summer ) * 100
diff_sum_05_obs_imerg_rl  = ((diff_sum_05_obs_imerg ) / cli_imerg_summer ) * 100

# Saving the Python Data

diff_sum_08_obs_imerg.to_netcdf("diff_sum_08_obs_imerg.nc")
diff_sum_06_obs_imerg.to_netcdf("diff_sum_06_obs_imerg.nc")
diff_sum_05_obs_imerg.to_netcdf("diff_sum_05_obs_imerg.nc")

# Relative difference between the Data

diff_sum_08_obs_imerg_rl.to_netcdf("diff_sum_08_obs_imerg_rl.nc")
diff_sum_06_obs_imerg_rl.to_netcdf("diff_sum_06_obs_imerg_rl.nc")
diff_sum_05_obs_imerg_rl.to_netcdf("diff_sum_05_obs_imerg_rl.nc")



## Significancw

hatch_08 = np.abs(np.log1p(cli_08_summer) - np.log1p(cli_imerg_summer)) > 1.96 * np.sqrt(
    (np.log1p(var_08_summer).std("time")/np.sqrt(len(var_08_summer.time)))**2 + 
    (np.log1p(var_imerg_summer).std("time")/np.sqrt(len(var_imerg_summer.time)))**2
)


hatch_06 = np.abs(np.log1p(cli_06_summer) - np.log1p(cli_imerg_summer)) > 1.96 * np.sqrt(
    (np.log1p(var_06_summer).std("time")/np.sqrt(len(var_06_summer.time)))**2 + 
    (np.log1p(var_imerg_summer).std("time")/np.sqrt(len(var_imerg_summer.time)))**2
)

hatch_05 = np.abs(np.log1p(cli_05_summer) - np.log1p(cli_imerg_summer)) > 1.96 * np.sqrt(
    (np.log1p(var_05_summer).std("time")/np.sqrt(len(var_05_summer.time)))**2 + 
    (np.log1p(var_imerg_summer).std("time")/np.sqrt(len(var_imerg_summer.time)))**2
)



hatch_08.to_netcdf("hatch_08_summer.nc")
hatch_06.to_netcdf("hatch_06_summer.nc")
hatch_05.to_netcdf("hatch_05_summer.nc")



###   CPC

# Absolute Percentage Differences

diff_sum_08_obs_cpc  = cli_08_summer - cli_cpc_summer
diff_sum_06_obs_cpc  = cli_06_summer - cli_cpc_summer
diff_sum_05_obs_cpc  = cli_05_summer - cli_cpc_summer

# Relative Differences

diff_sum_08_obs_cpc_rl  = ((diff_sum_08_obs_cpc ) / cli_cpc_summer ) * 100
diff_sum_06_obs_cpc_rl  = ((diff_sum_06_obs_cpc ) / cli_cpc_summer ) * 100
diff_sum_05_obs_cpc_rl  = ((diff_sum_05_obs_cpc ) / cli_cpc_summer ) * 100

# Saving the Python Data

diff_sum_08_obs_cpc.to_netcdf("diff_sum_08_obs_cpc.nc")
diff_sum_06_obs_cpc.to_netcdf("diff_sum_06_obs_cpc.nc")
diff_sum_05_obs_cpc.to_netcdf("diff_sum_05_obs_cpc.nc")

# Relative difference between the Data

diff_sum_08_obs_cpc_rl.to_netcdf("diff_sum_08_obs_cpc_rl.nc")
diff_sum_06_obs_cpc_rl.to_netcdf("diff_sum_06_obs_cpc_rl.nc")
diff_sum_05_obs_cpc_rl.to_netcdf("diff_sum_05_obs_cpc_rl.nc")


### MSWEP

# Absolute Percentage Differences

diff_sum_08_obs_mswep  = cli_08_summer - cli_mswep_summer
diff_sum_06_obs_mswep  = cli_06_summer - cli_mswep_summer
diff_sum_05_obs_mswep  = cli_05_summer - cli_mswep_summer

# Relative Differences

diff_sum_08_obs_mswep_rl  = ((diff_sum_08_obs_mswep ) / cli_mswep_summer ) * 100
diff_sum_06_obs_mswep_rl  = ((diff_sum_06_obs_mswep ) / cli_mswep_summer ) * 100
diff_sum_05_obs_mswep_rl  = ((diff_sum_05_obs_mswep ) / cli_mswep_summer ) * 100

# Saving the Python Data

diff_sum_08_obs_mswep.to_netcdf("diff_sum_08_obs_mswep.nc")
diff_sum_06_obs_mswep.to_netcdf("diff_sum_06_obs_mswep.nc")
diff_sum_05_obs_mswep.to_netcdf("diff_sum_05_obs_mswep.nc")

# Relative difference between the Data

diff_sum_08_obs_mswep_rl.to_netcdf("diff_sum_08_obs_mswep_rl.nc")
diff_sum_06_obs_mswep_rl.to_netcdf("diff_sum_06_obs_mswep_rl.nc")
diff_sum_05_obs_mswep_rl.to_netcdf("diff_sum_05_obs_mswep_rl.nc")


### ERA5

# Absolute Percentage Differences

diff_sum_08_obs_era5  = cli_08_summer - cli_era5_summer
diff_sum_06_obs_era5  = cli_06_summer - cli_era5_summer
diff_sum_05_obs_era5  = cli_05_summer - cli_era5_summer

# Relative Differences

diff_sum_08_obs_era5_rl  = ((diff_sum_08_obs_era5 ) / cli_era5_summer ) * 100
diff_sum_06_obs_era5_rl  = ((diff_sum_06_obs_era5 ) / cli_era5_summer ) * 100
diff_sum_05_obs_era5_rl  = ((diff_sum_05_obs_era5 ) / cli_era5_summer ) * 100

# Saving the Python Data

diff_sum_08_obs_era5.to_netcdf("diff_sum_08_obs_era5.nc")
diff_sum_06_obs_era5.to_netcdf("diff_sum_06_obs_era5.nc")
diff_sum_05_obs_era5.to_netcdf("diff_sum_05_obs_era5.nc")

# Relative difference between the Data

diff_sum_08_obs_era5_rl.to_netcdf("diff_sum_08_obs_era5_rl.nc")
diff_sum_06_obs_era5_rl.to_netcdf("diff_sum_06_obs_era5_rl.nc")
diff_sum_05_obs_era5_rl.to_netcdf("diff_sum_05_obs_era5_rl.nc")


### CMORPH

# Absolute Percentage Differences

diff_sum_08_obs_cmorph  = cli_08_summer - cli_cmorph_summer
diff_sum_06_obs_cmorph  = cli_06_summer - cli_cmorph_summer
diff_sum_05_obs_cmorph  = cli_05_summer - cli_cmorph_summer

# Relative Differences

diff_sum_08_obs_cmorph_rl  = ((diff_sum_08_obs_cmorph ) / cli_cmorph_summer ) * 100
diff_sum_06_obs_cmorph_rl  = ((diff_sum_06_obs_cmorph ) / cli_cmorph_summer ) * 100
diff_sum_05_obs_cmorph_rl  = ((diff_sum_05_obs_cmorph ) / cli_cmorph_summer ) * 100

# Saving the Python Data

diff_sum_08_obs_cmorph.to_netcdf("diff_sum_08_obs_cmorph.nc")
diff_sum_06_obs_cmorph.to_netcdf("diff_sum_06_obs_cmorph.nc")
diff_sum_05_obs_cmorph.to_netcdf("diff_sum_05_obs_cmorph.nc")

# Relative difference between the Data

diff_sum_08_obs_cmorph_rl.to_netcdf("diff_sum_08_obs_cmorph_rl.nc")
diff_sum_06_obs_cmorph_rl.to_netcdf("diff_sum_06_obs_cmorph_rl.nc")
diff_sum_05_obs_cmorph_rl.to_netcdf("diff_sum_05_obs_cmorph_rl.nc")



