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
##############################################################################
#### Namelist (all the user specified settings at the start of the code
####           separated from the rest of the code)
##############################################################################

#Here choose NH_Summer or NH_Winter or NH_Summer-NH_Winter
season='NH_Summer'

# base directory where your analysis data is stored GPU
# Directory of 10KM simulation 
data_base_dir_08 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/tot_prec'
model_data_08    =  'tot_prec_30_day_land.nc'

# Directoy for 40KM simulation (Time step = 4018)
data_base_dir_06 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/tot_prec'
model_data_06    =  'tot_prec_30_day_land.nc'

# Directory for 80KM simulation (Time step = 4018)
data_base_dir_05 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/tot_prec'
model_data_05    =  'tot_prec_30_day_land.nc'

# Observational Data Sets Starts from Here. 

# # Observational Data Sets for IMERG (Time step = 3653)
data_base_dir_imerg = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/IMERG/day_nc4_files/post_processed_files'
model_data_imerg    =  'precipitation_cdo_all_rmp_10years_land.nc'

# Observational Data Sets for CPC (Time step = 3653)

data_base_dir_cpc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CPC_data/daily'
model_data_cpc    =  'precip.daily_rmp_10years.nc'

# Observational Data Sets for ERA5 (Time step = 3653)

data_base_dir_era5 = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/ERA5_Data/daily'
model_data_era5    =  'pr_day_reanalysis_era5_r1i1p1_daily_rmp_10years_land.nc'

# Obesrvational Data Set for MSWEP (Time step = 3653)

data_base_dir_mswep = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/MSWEP/Data_2006_2022'
model_data_mswep    = 'precip_rmp_10years_land.nc'

# CMORPH Data ((Time step = 3653))

data_base_dir_cmorph = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/CMORPH/data/CMORPH/daily_nc'
model_data_cmorph    = 'daily_rmp_10years_land.nc'

# GPCC Data ((Time step = 3653) 

data_base_dir_gpcc = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/GPCC_data/daily'
model_data_gpcc    = 'full_data_rmp_10years.nc'


## LAND MASK
data_base_dir_gpcc_mask = '/capstor/store/cscs/userlab/cwp03/ppothapa/data_from_excp01/Monsoon_Final/obervations/GPCC_data/daily'
model_data_gpcc_mask    = 'mask_precip_gpcc.nc'



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
model_data_dir_gpcc_mask    = os.path.join(data_base_dir_gpcc_mask, model_data_gpcc_mask)



ds_08  = xr.open_dataset(model_data_dir_08)
ds_06  = xr.open_dataset(model_data_dir_06)
ds_05  = xr.open_dataset(model_data_dir_05)
ds_imerg  = xr.open_dataset(model_data_dir_imerg)
ds_cpc  = xr.open_dataset(model_data_dir_cpc)
ds_era5  = xr.open_dataset(model_data_dir_era5)
ds_mswep  = xr.open_dataset(model_data_dir_mswep)
ds_cmorph  = xr.open_dataset(model_data_dir_cmorph)
ds_gpcc  = xr.open_dataset(model_data_dir_gpcc)
ds_gpcc_mask  = xr.open_dataset(model_data_dir_gpcc_mask)



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
bounds = np.arange(55,105,5) # Here the limit is till 55 to 100, it is strange in python
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
     var_gpcc_mask = ds_gpcc_mask['precip']

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
     var_gpcc_mask = var_gpcc_mask.sel(lat=slice(-40,40))

###  Here Multiply the Var values. 

#     var_08 = var_08 * var_gpcc_mask
#     var_06 = var_06 * var_gpcc_mask
#     var_05 = var_05 * var_gpcc_mask
#     var_imerg = var_imerg * var_gpcc_mask   
#     var_era5  = var_era5 * var_gpcc_mask
#     var_mswep  = var_mswep * var_gpcc_mask
#     var_cmorph  = var_cmorph * var_gpcc_mask

## Annual sum of every year of the simulations. Used to calculate the mask for 55%.

     evolve_var_08_annual  = var_08.resample(time='YE').sum()
     print('shape of R02B08 is:') 
     print(evolve_var_08_annual.shape)

     evolve_var_06_annual  = var_06.resample(time='YE').sum()
     print('shape of R02B06 is:')
     print(evolve_var_06_annual.shape)

     evolve_var_05_annual  = var_05.resample(time='YE').sum()
     print('shape of R02B05 is:')
     print(evolve_var_05_annual.shape)

     evolve_var_era5_annual = var_era5.resample(time='YE').sum() * 86400 
     print('shape of ERA5 is:')
     print(evolve_var_era5_annual.shape)

     evolve_var_imerg_annual = var_imerg.resample(time='YE').sum()
     print('shape of IMERG is:')
     print(evolve_var_imerg_annual.shape)

     evolve_var_cpc_annual = var_cpc.resample(time='YE').sum() 
     print('shape of CPC is:')
     print(evolve_var_cpc_annual.shape)

     evolve_var_mswep_annual = var_mswep.resample(time='YE').sum() 
     print('shape of MSWEP is:')
     print(evolve_var_mswep_annual.shape)

     evolve_var_cmorph_annual = var_cmorph.resample(time='YE').sum()
     print('shape of CMORPH is:')
     print(evolve_var_cmorph_annual.shape)

     evolve_var_gpcc_annual = var_gpcc.resample(time='YE').sum()
     print('shape of GPCC is:')
     print(evolve_var_gpcc_annual.shape)

# This is the annual mean of the precipitation

     evolve_var_08_annual_mean 	= evolve_var_08_annual.mean('time')
     evolve_var_06_annual_mean 	= evolve_var_06_annual.mean('time')
     evolve_var_05_annual_mean	 = evolve_var_05_annual.mean('time')
     evolve_var_imerg_annual_mean = evolve_var_imerg_annual.mean('time')
     evolve_var_era5_annual_mean = evolve_var_era5_annual.mean('time')
     evolve_var_cpc_annual_mean = evolve_var_cpc_annual.mean('time')
     evolve_var_mswep_annual_mean = evolve_var_mswep_annual.mean('time')
     evolve_var_cmorph_annual_mean = evolve_var_cmorph_annual.mean('time')
     evolve_var_gpcc_annual_mean = evolve_var_gpcc_annual.mean('time')


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



#  Calculate the Summer & winter contribution (mm/season) climatology

     evolve_var_08_su  = var_08_summer.resample(time='YE').sum().mean("time")
     evolve_var_08_win = var_08_winter.resample(time='YE').sum().mean("time")
    
     evolve_var_06_su  = var_06_summer.resample(time='YE').sum().mean("time")
     evolve_var_06_win = var_06_winter.resample(time='YE').sum().mean("time") 

     evolve_var_05_su  = var_05_summer.resample(time='YE').sum().mean("time")
     evolve_var_05_win = var_05_winter.resample(time='YE').sum().mean("time")

     evolve_var_era5_su  = var_era5_summer.resample(time='YE').sum().mean("time") * 86400
     evolve_var_era5_win = var_era5_winter.resample(time='YE').sum().mean("time") * 86400

     evolve_var_imerg_su  = var_imerg_summer.resample(time='YE').sum().mean("time") 
     evolve_var_imerg_win = var_imerg_winter.resample(time='YE').sum().mean("time") 

     evolve_var_cpc_su  = var_cpc_summer.resample(time='YE').sum().mean("time") 
     evolve_var_cpc_win = var_cpc_winter.resample(time='YE').sum().mean("time") 

     evolve_var_mswep_su  = var_mswep_summer.resample(time='YE').sum().mean("time") 
     evolve_var_mswep_win = var_mswep_winter.resample(time='YE').sum().mean("time") 
     
     evolve_var_cmorph_su  = var_cmorph_summer.resample(time='YE').sum().mean("time")
     evolve_var_cmorph_win = var_cmorph_winter.resample(time='YE').sum().mean("time")

     evolve_var_gpcc_su  = var_gpcc_summer.resample(time='YE').sum().mean("time")
     evolve_var_gpcc_win = var_gpcc_winter.resample(time='YE').sum().mean("time")



# Calulate Percentage contribution for the Summer.

     percentage_contribution_sum_08 = 100 * (evolve_var_08_su/evolve_var_08_annual_mean)
     percentage_contribution_sum_06 = 100 * (evolve_var_06_su/evolve_var_06_annual_mean)
     percentage_contribution_sum_05 = 100 * (evolve_var_05_su/evolve_var_05_annual_mean)
     percentage_contribution_sum_era5  = 100 * (evolve_var_era5_su/evolve_var_era5_annual_mean)
     percentage_contribution_sum_imerg = 100 * (evolve_var_imerg_su/evolve_var_imerg_annual_mean)
     percentage_contribution_sum_cpc   = 100 * (evolve_var_cpc_su/evolve_var_cpc_annual_mean)
     percentage_contribution_sum_mswep = 100 * (evolve_var_mswep_su/evolve_var_mswep_annual_mean)
     percentage_contribution_sum_cmorph = 100 * (evolve_var_cmorph_su/evolve_var_cmorph_annual_mean)
     percentage_contribution_sum_gpcc = 100 * (evolve_var_gpcc_su/evolve_var_gpcc_annual_mean)
   
# Calculate Percentage contribution for the winter

    
     percentage_contribution_win_08 = 100 * (evolve_var_08_win/evolve_var_08_annual_mean)
     percentage_contribution_win_06 = 100 * (evolve_var_06_win/evolve_var_06_annual_mean)
     percentage_contribution_win_05 = 100 * (evolve_var_05_win/evolve_var_05_annual_mean)
     percentage_contribution_win_era5 = 100 * (evolve_var_era5_win/evolve_var_era5_annual_mean)
     percentage_contribution_win_imerg = 100 * (evolve_var_imerg_win/evolve_var_imerg_annual_mean)
     percentage_contribution_win_cpc = 100 * (evolve_var_cpc_win/evolve_var_cpc_annual_mean)
     percentage_contribution_win_mswep = 100 * (evolve_var_mswep_win/evolve_var_mswep_annual_mean)
     percentage_contribution_win_cmorph = 100 * (evolve_var_cmorph_win/evolve_var_cmorph_annual_mean)
     percentage_contribution_win_gpcc = 100 * (evolve_var_gpcc_win/evolve_var_gpcc_annual_mean)


     # NH summer-dominant mask (>55% contribution)
     NH_mask_08 = ((percentage_contribution_sum_08 > 55) & (var_08_summer_winter > 2) & (var_08_summer_winter.lat > 0)).astype(int)
     NH_mask_06 = ((percentage_contribution_sum_06 > 55) & (var_06_summer_winter > 2) & (var_06_summer_winter.lat > 0)).astype(int)
     NH_mask_05 = ((percentage_contribution_sum_05 > 55) & (var_05_summer_winter > 2) & (var_05_summer_winter.lat > 0)).astype(int)
     NH_mask_era5 = ((percentage_contribution_sum_era5 > 55) & (var_era5_summer_winter > 2) & (var_era5_summer_winter.lat > 0)).astype(int)
     NH_mask_imerg = ((percentage_contribution_sum_imerg > 55) & (var_imerg_summer_winter > 2) & (var_imerg_summer_winter.lat > 0)).astype(int)
     NH_mask_cpc = ((percentage_contribution_sum_cpc > 55) & (var_cpc_summer_winter > 2) & (var_cpc_summer_winter.lat > 0)).astype(int)
     NH_mask_mswep = ((percentage_contribution_sum_mswep > 55) & (var_mswep_summer_winter > 2) & (var_mswep_summer_winter.lat > 0)).astype(int)
     NH_mask_cmorph = ((percentage_contribution_sum_cmorph > 55) & (var_cmorph_summer_winter > 2) & (var_cmorph_summer_winter.lat > 0)).astype(int)
     NH_mask_gpcc = ((percentage_contribution_sum_gpcc > 55) & (var_gpcc_summer_winter > 2) & (var_gpcc_summer_winter.lat > 0)).astype(int)


   # SH summer-dominant mask (>55% winter contribution)
     SH_mask_08 = ((percentage_contribution_win_08 > 55) & (var_08_summer_winter < -2) &  (var_08_summer_winter.lat < 0)).astype(int)
     SH_mask_06 = ((percentage_contribution_win_06 > 55) & (var_06_summer_winter < -2) &  (var_06_summer_winter.lat < 0)).astype(int)
     SH_mask_05 = ((percentage_contribution_win_05 > 55) & (var_05_summer_winter < -2) &  (var_05_summer_winter.lat < 0)).astype(int)
     SH_mask_era5 = ((percentage_contribution_win_era5 > 55) & (var_era5_summer_winter < -2) &  (var_era5_summer_winter.lat < 0)).astype(int)
     SH_mask_imerg = ((percentage_contribution_win_imerg > 55) & (var_imerg_summer_winter < -2) & (var_imerg_summer_winter.lat < 0)).astype(int)
     SH_mask_cpc = ((percentage_contribution_win_cpc > 55) & (var_cpc_summer_winter < -2) & (var_cpc_summer_winter.lat < 0)).astype(int)
     SH_mask_mswep = ((percentage_contribution_win_mswep > 55) & (var_mswep_summer_winter < -2) & (var_mswep_summer_winter.lat < 0)).astype(int)
     SH_mask_cmorph = ((percentage_contribution_win_cmorph > 55) & (var_cmorph_summer_winter < -2) & (var_cmorph_summer_winter.lat < 0)).astype(int)
     SH_mask_gpcc = ((percentage_contribution_win_gpcc > 55) & (var_gpcc_summer_winter < -2) & (var_gpcc_summer_winter.lat < 0)).astype(int)

     # --- Step 3: Combine NH and SH masks ---
     monsoon_mask_08 = NH_mask_08 + SH_mask_08  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_06 = NH_mask_06 + SH_mask_06  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_05 = NH_mask_05 + SH_mask_05  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_era5 = NH_mask_era5 + SH_mask_era5  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_imerg = NH_mask_imerg + SH_mask_imerg  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_cpc = NH_mask_cpc + SH_mask_cpc  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_mswep = NH_mask_mswep + SH_mask_mswep  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_cmorph = NH_mask_cmorph + SH_mask_cmorph  # 1 for monsoon regions, 0 elsewhere
     monsoon_mask_gpcc = NH_mask_gpcc + SH_mask_gpcc  # 1 for monsoon regions, 0 elsewhere

     # Optional: make sure mask is exactly 0/1
     monsoon_mask_08 = monsoon_mask_08.where(monsoon_mask_08 > 0, 0)
     monsoon_mask_06 = monsoon_mask_06.where(monsoon_mask_06 > 0, 0)
     monsoon_mask_05 = monsoon_mask_05.where(monsoon_mask_05 > 0, 0)     
     monsoon_mask_era5 = monsoon_mask_era5.where(monsoon_mask_era5 > 0, 0)
     monsoon_mask_imerg = monsoon_mask_imerg.where(monsoon_mask_imerg > 0, 0)
     monsoon_mask_cpc = monsoon_mask_cpc.where(monsoon_mask_cpc > 0, 0)
     monsoon_mask_mswep = monsoon_mask_mswep.where(monsoon_mask_mswep > 0, 0)
     monsoon_mask_cmorph = monsoon_mask_cmorph.where(monsoon_mask_cmorph > 0, 0)
     monsoon_mask_gpcc = monsoon_mask_gpcc.where(monsoon_mask_gpcc > 0, 0)


# Now Save the Mask for the 2mm/day as well as 55 % Values

     monsoon_mask_08.to_netcdf("08_mask_2mm_55.nc")
     monsoon_mask_06.to_netcdf("06_mask_2mm_55.nc")
     monsoon_mask_05.to_netcdf("05_mask_2mm_55.nc")
     monsoon_mask_era5.to_netcdf("era5_mask_2mm_55.nc")
     monsoon_mask_imerg.to_netcdf("imerg_mask_2mm_55.nc")
     monsoon_mask_cpc.to_netcdf("cpc_mask_2mm_55.nc")
     monsoon_mask_mswep.to_netcdf("mswep_mask_2mm_55.nc")
     monsoon_mask_cmorph.to_netcdf("cmorph_mask_2mm_55.nc")
     monsoon_mask_gpcc.to_netcdf("gpcc_mask_2mm_55.nc")
     

#  Preparing for the figures & Spceifying the Projections
     proj = ccrs.Mercator()
     fig = plt.figure(layout='constrained',figsize=(12, 12))
     # set the spacing between subplots
     plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.1)
#     fig.tight_layout() 
#  Plotting the figures with a 2X2 Matrix
     # Subplot for the Top Left
   
     ax1 = fig.add_subplot(3,3,1, projection=proj)     
     if season=='NH_Summer':
         plot = percentage_contribution_both_08.plot(
                      ax=ax1,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'}, 
                     )
     # Add coast line 
         ax1.coastlines(resolution='10m', lw=0.51)
         ax1.set_title(f'ICON-10KM (NH Summer Contribution)')
        # ax1.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
         ax1.set_aspect(2)
     elif season=='NH_Winter':
         plot = percentage_contribution_win_08_1.plot(
                      ax=ax1,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
         ax1.coastlines(resolution='10m', lw=0.51)
         ax1.set_aspect(2)
         ax1.set_title(f'ICON-10KM (SH Summer Contribution)')
     #    ax1.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     elif season=='NH_Summer_NH_Winter':
         plot = var_08_summer_winter.plot(
                      ax=ax1,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
    
         
         plot = var_08_summer_winter_NH_only.plot.contour(ax=ax1,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
         plot = var_08_summer_winter_SH_only.plot.contour(ax=ax1,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')


         # Add coast line
         ax1.coastlines(resolution='10m', lw=0.51)
         ax1.set_title(f'ICON-10KM (Summer-Winter)')
         ax1.set_aspect(2)
#         ax1.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     gls = ax1.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,color="none",linewidth=2,alpha=0.5, linestyle='--')
     gls.top_labels=False
     gls.right_labels=False
#     _ = fig.subplots_adjust(left=0.2, right=0.8, hspace=0, wspace=0, top=0.8, bottom=0.25)

#   Figure 2
     ax2 = fig.add_subplot(3,3,2, projection=proj)
     if season=='NH_Summer':
         plot = percentage_contribution_both_06.plot(
                      ax=ax2,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
 #                     vmin=55,
 #                     vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
         ax2.coastlines(resolution='10m', lw=0.51)
         ax2.set_title(f'ICON-40KM (NH Summer Contribution)')
     #    ax2.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
         ax2.set_aspect(2)
     elif season=='NH_Winter':
         plot = percentage_contribution_win_06_1.plot(
                      ax=ax2,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )

     # Add coast line
         ax2.coastlines(resolution='10m', lw=0.51)
         ax2.set_title(f'ICON-40KM (SH Summer Contribution)')
     #    ax2.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
         ax2.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
         plot = var_06_summer_winter.plot(
                      ax=ax2,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
         
         plot = var_06_summer_winter_NH_only.plot.contour(ax=ax2,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
         plot = var_06_summer_winter_SH_only.plot.contour(ax=ax2,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')


         ax2.coastlines(resolution='10m', lw=0.51)
         ax2.set_title(f'ICON-40KM (Summer-Winter)')
 #        ax2.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
         ax2.set_aspect(2)

    
     gls = ax2.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False

#    Figure 3
     ax3 = fig.add_subplot(3,3,3, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_05.plot(
                      ax=ax3,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax3.coastlines(resolution='10m', lw=0.51)
        ax3.set_title(f'ICON-80KM (NH Summer Contribution)')
    #    ax3.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax3.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_05_1.plot(
                      ax=ax3,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )

     # Add coast line
        ax3.coastlines(resolution='10m', lw=0.51)
        ax3.set_title(f'ICON-80KM (SH Winter Contribution)')
     #   ax3.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax3.set_aspect(2)
 
     elif season=='NH_Summer_NH_Winter':
          plot = var_05_summer_winter.plot(
                      ax=ax3,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
          
          plot = var_05_summer_winter_NH_only.plot.contour(ax=ax3,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
          plot = var_05_summer_winter_SH_only.plot.contour(ax=ax3,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')


     # Add coast line
          ax3.coastlines(resolution='10m', lw=0.51)
          ax3.set_title(f'ICON-80KM (Summer-Winter)')
 #         ax3.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

          ax3.set_aspect(2)

     gls = ax3.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False


# Figure 4 

     ax4 = fig.add_subplot(3,3,4, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_era5.plot(
                      ax=ax4,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax4.coastlines(resolution='10m', lw=0.51)
        ax4.set_title(f'ERA5 (NH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax4.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_era5_1.plot(
                      ax=ax4,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax4.coastlines(resolution='10m', lw=0.51)
        ax4.set_title(f'ERA5 (SH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax4.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
        plot = var_era5_summer_winter.plot(
                      ax=ax4,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )


        
        plot = var_era5_summer_winter_NH_only.plot.contour(ax=ax4,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
        plot = var_era5_summer_winter_SH_only.plot.contour(ax=ax4,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
     # Add coast line
        ax4.coastlines(resolution='10m', lw=0.51)
        ax4.set_title(f'ERA5 (Summer-Winter)')
        ax4.set_aspect(2)
#        ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
     
     gls = ax4.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False

   
     # Figure 5

     ax5 = fig.add_subplot(3,3,5, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_imerg.plot(
                      ax=ax5,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax5.coastlines(resolution='10m', lw=0.51)
        ax5.set_title(f'IMERG (NH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax5.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_imerg_1.plot(
                      ax=ax5,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax4.coastlines(resolution='10m', lw=0.51)
        ax4.set_title(f'IMERG (SH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax4.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
        plot = var_imerg_summer_winter.plot(
                      ax=ax5,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )



        plot = var_imerg_summer_winter_NH_only.plot.contour(ax=ax5,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
        plot = var_imerg_summer_winter_SH_only.plot.contour(ax=ax5,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
     # Add coast line
        ax5.coastlines(resolution='10m', lw=0.51)
        ax5.set_title(f'IMERG (Summer-Winter)')
        ax5.set_aspect(2)
#        ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     gls = ax5.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False




     # Figure 6

     ax6 = fig.add_subplot(3,3,6, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_cpc.plot(
                      ax=ax6,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax6.coastlines(resolution='10m', lw=0.51)
        ax6.set_title(f'CPC (NH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax6.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_cpc_1.plot(
                      ax=ax6,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax6.coastlines(resolution='10m', lw=0.51)
        ax6.set_title(f'CPC (SH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax6.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
        plot = var_cpc_summer_winter.plot(
                      ax=ax6,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )



        plot = var_cpc_summer_winter_NH_only.plot.contour(ax=ax6,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
        plot = var_cpc_summer_winter_SH_only.plot.contour(ax=ax6,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
     # Add coast line
        ax6.coastlines(resolution='10m', lw=0.51)
        ax6.set_title(f'CPC (Summer-Winter)')
        ax6.set_aspect(2)
#        ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     gls = ax6.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False




     # Figure 7

     ax7 = fig.add_subplot(3,3,7, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_gpcc.plot(
                      ax=ax7,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax7.coastlines(resolution='10m', lw=0.51)
        ax7.set_title(f'GPCC (NH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax7.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_gpcc_1.plot(
                      ax=ax7,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax7.coastlines(resolution='10m', lw=0.51)
        ax7.set_title(f'ERA5 (SH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax7.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
        plot = var_gpcc_summer_winter.plot(
                      ax=ax7,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )



        plot = var_gpcc_summer_winter_NH_only.plot.contour(ax=ax7,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
        plot = var_gpcc_summer_winter_SH_only.plot.contour(ax=ax7,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
     # Add coast line
        ax7.coastlines(resolution='10m', lw=0.51)
        ax7.set_title(f'GPCC (Summer-Winter)')
        ax7.set_aspect(2)
#        ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     gls = ax7.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False



 # Figure 8

     ax8 = fig.add_subplot(3,3,8, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_mswep.plot(
                          ax=ax8,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax8.coastlines(resolution='10m', lw=0.51)
        ax8.set_title(f'MSWEP (NH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax8.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_gpcc_1.plot(
                      ax=ax8,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax8.coastlines(resolution='10m', lw=0.51)
        ax8.set_title(f'MSWEP (SH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax8.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
        plot = var_mswep_summer_winter.plot(
                      ax=ax8,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )



        plot = var_mswep_summer_winter_NH_only.plot.contour(ax=ax8,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
        plot = var_mswep_summer_winter_SH_only.plot.contour(ax=ax8,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
     # Add coast line
        ax8.coastlines(resolution='10m', lw=0.51)
        ax8.set_title(f'MSWEP (Summer-Winter)')
        ax8.set_aspect(2)
#        ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     gls = ax8.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False



# Figure 9

 # Figure 7

     ax9 = fig.add_subplot(3,3,9, projection=proj)
     if season=='NH_Summer':
        plot = percentage_contribution_both_cmorph.plot(
                      ax=ax9,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      norm=norm,
#                      vmin=55,
#                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax9.coastlines(resolution='10m', lw=0.51)
        ax9.set_title(f'CMORP (NH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax9.set_aspect(2)

     elif season=='NH_Winter':
        plot = percentage_contribution_win_cmorph_1.plot(
                      ax=ax9,
                      transform=ccrs.PlateCarree(),
                      cmap='Greens',
                      vmin=55,
                      vmax=100,
                      cbar_kwargs={'label': '% Contribution of Total Precipitation',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )
     # Add coast line
        ax9.coastlines(resolution='10m', lw=0.51)
        ax9.set_title(f'CMORPH (SH Summer Contribution)')
   #     ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))
        ax4.set_aspect(2)

     elif season=='NH_Summer_NH_Winter':
        plot = var_cmorph_summer_winter.plot(
                      ax=ax9,
                      transform=ccrs.PlateCarree(),
                      cmap='bwr',
                      vmin=-10,
                      vmax=10,
                      cbar_kwargs={'label': 'Total Precipitation (mm/day)',
                                   'extend': 'both',
                                   'shrink': .6,
                                   'orientation': 'horizontal'},
                     )



        plot = var_cmorph_summer_winter_NH_only.plot.contour(ax=ax9,levels=[2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
        plot = var_cmorph_summer_winter_SH_only.plot.contour(ax=ax9,levels=[-2], transform=ccrs.PlateCarree(), colors=['m'],linestyles='-')
     # Add coast line
        ax9.coastlines(resolution='10m', lw=0.51)
        ax9.set_title(f'CMORPH (Summer-Winter)')
        ax9.set_aspect(2)
#        ax4.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k',facecolor=(1,1,1))

     gls = ax9.gridlines(draw_labels=True,color="none")
     gls.top_labels=False
     gls.right_labels=False




#Save the Figure in png format
     plt.savefig(plot_name)
