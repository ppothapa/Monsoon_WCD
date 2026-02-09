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

from cdo import *
cdo = Cdo()

tempPath = "/capstor/scratch/cscs/ppothapa/cdo_temp_dir"
cdo = Cdo(tempdir=tempPath)


#Here choose NH_Summer or NH_Winter or NH_Summer-NH_Winter
season='NH_Summer'

Region="Indian_Core"
plot_name="Indian_Core"
plot_var_key="tot_prec"

fig = plt.figure(figsize=(12, 10), constrained_layout=True)

reg = 'South_Asia','North Africa','North America'

# base directory where your analysis data is stored GPU
# Directory of 10KM simulation
data_base_dir_08 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B08L120/tot_prec'
model_data_08    =  'tot_prec_30.nc'

# Directoy for 40KM simulation (Time step = 4018)
data_base_dir_06 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B06L120/tot_prec'
model_data_06    =  'tot_prec_30.nc'

# Directory for 80KM simulation (Time step = 4018)
data_base_dir_05 = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/ICON_Model_Data/exclaim_uncoupled_R02B05L120/tot_prec'
model_data_05    =  'tot_prec_30.nc'

# # Observational Data Sets for IMERG (Time step = 4017)

data_base_dir_imerg = '/capstor/store/cscs/exclaim/excp01/ppothapa/Monsoon_Final/obervations/IMERG/sub_daily/Final_Data/merged'
model_data_imerg    =  'IMERG_2007_2016_hourly.nc'


for x in reg:

    if Region == 'Indian_Core':
    # Select a Region
    #South Asia
       lat_min=18
       lat_max=25
       lon_min=72
       lon_max=85
       plot_name='Indian_Core'

    elif Region == 'North Africa':
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

    elif Region == 'North America':
         lat_min=10
         lat_max=20
         lon_min=-118
         lon_max=-90
         plot_name='North_America'


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


# Special Operationfor IMERG

    var_imerg = var_imerg.transpose()

# Select Lat & Lon focussing on the tropics 
    var_08 = var_08.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_06 = var_06.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_05 = var_05.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    var_imerg = var_imerg.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))

# We calculate Only Norther Hemisphere Interannual Variability

    var_08_summer = var_08.sel(time=var_08.time.dt.month.isin([6, 7, 8, 9]))
    var_08_winter = var_08.sel(time=var_08.time.dt.month.isin([12, 1, 2, 3]))

    var_06_summer = var_06.sel(time=var_06.time.dt.month.isin([6, 7, 8, 9]))
    var_06_winter = var_06.sel(time=var_06.time.dt.month.isin([12, 1, 2, 3]))

    var_05_summer = var_05.sel(time=var_05.time.dt.month.isin([6, 7, 8, 9]))
    var_05_winter = var_05.sel(time=var_05.time.dt.month.isin([12, 1, 2, 3]))

    var_imerg_summer = var_imerg.sel(time=var_imerg.time.dt.month.isin([6, 7, 8, 9]))
    var_imerg_winter = var_imerg.sel(time=var_imerg.time.dt.month.isin([12, 1, 2, 3]))


# Do CDO dhour mean 

    var_08_summer_diurnal = cdo.dhourmean(input=var_08_summer, options="-P 24")

    var_06_summer_diurnal = cdo.dhourmean(input=var_06_summer, options="-P 24")

    var_05_summer_diurnal = cdo.dhourmean(input=var_05_summer, options="-P 24")

    var_imerg_summer_diurnal = cdo.dhourmean(input=var_imerg_summer, options="-P 24")


# Do CDO Fldmean 

    var_08_summer_diurnal_fldmean = cdo.fldmean(input=var_08_summer_diurnal,  options="-P 24")
    var_06_summer_diurnal_fldmean = cdo.fldmean(input=var_06_summer_diurnal,  options="-P 24")
    var_05_summer_diurnal_fldmean = cdo.fldmean(input=var_05_summer_diurnal,  options="-P 24")
    var_imerg_summer_diurnal_fldmean = cdo.fldmean(input=var_imerg_summer_diurnal,  options="-P 24")


## Open the dataset into python again because CDO saves times in netcdf format temporarly in the    

    cycle_08 = xr.open_dataset(var_08_summer_diurnal_fldmean)
    cycle_06 = xr.open_dataset(var_06_summer_diurnal_fldmean)
    cycle_05 = xr.open_dataset(var_05_summer_diurnal_fldmean)
    cycle_imerg = xr.open_dataset(var_imerg_summer_diurnal_fldmean)

## Open the variable

    cycle_08_var = cycle_08[plot_var_key]
    cycle_06_var = cycle_06[plot_var_key] 
    cycle_05_var = cycle_05[plot_var_key]

## Conver Xarry format to the nunmpy format. 

    cycle_08_var_numpy = cycle_08_var.to_numpy()
    cycle_06_var_numpy = cycle_06_var.to_numpy()
    cycle_05_var_numpy = cycle_05_var.to_numpy()

#Reshape Xarray

    cycle_08_var_numpy_vector = cycle_08_var_numpy.reshape(-1)
    cycle_06_var_numpy_vector = cycle_06_var_numpy.reshape(-1)
    cycle_05_var_numpy_vector = cycle_05_var_numpy.reshape(-1)
    cycle_imerg_var_numpy_vector = cycle_imerg_var_numpy.reshape(-1)


## Rearrange From 01 to 24

    cycle_08_var_numpy_vector_1to4 = cycle_08_var_numpy_vector[20:24]
    cycle_06_var_numpy_vector_1to4 = cycle_06_var_numpy_vector[20:24]
    cycle_05_var_numpy_vector_1to4 = cycle_05_var_numpy_vector[20:24]
    cycle_imerg_var_numpy_vector_1to4 = cycle_imerg_var_numpy_vector[20:24]


    cycle_08_var_numpy_vector_5to23 = cycle_08_var_numpy_vector[0:20]
    cycle_06_var_numpy_vector_5to23 = cycle_06_var_numpy_vector[0:20]
    cycle_05_var_numpy_vector_5to23 = cycle_05_var_numpy_vector[0:20]
    cycle_imerg_var_numpy_vector_5to23 = cycle_imerg_var_numpy_vector[0:20]
##

    cycle_08_var_adjusted  = np.concatenate((cycle_08_var_numpy_vector_1to4,cycle_08_var_numpy_vector_5to23))
    cycle_06_var_adjusted  = np.concatenate((cycle_06_var_numpy_vector_1to4,cycle_06_var_numpy_vector_5to23))
    cycle_05_var_adjusted  = np.concatenate((cycle_05_var_numpy_vector_1to4,cycle_05_var_numpy_vector_5to23))
    cycle_imerg_var_adjusted  = np.concatenate((cycle_imerg_var_numpy_vector_1to4,cycle_imerg_var_numpy_vector_5to23))

##

 
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)

    xmarks=["05:30","06:30","07:30","08:30", "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30" , "16:30", "17:30", "18:30", "19:30", "20:30", "21:30", "22:30", "23:30", "00:30", "01:30", "02:30", "03:30", "04:30"]

    xmarks =["1","2","3", "4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"]

    plt.plot(xmarks,cycle_08_var_adjusted, color="red", label="ICON (10KM)",  linewidth=2.5)
    plt.plot(xmarks,cycle_06_var_adjusted, color="green", label="ICON (40KM)",linewidth=2.5)
    plt.plot(xmarks,cycle_05_var_adjusted, color="blue", label="ICON (80KM)",linewidth=2.5)
    plt.plot(xmarks,cycle_imerg_var_adjusted, color="black", label="IMERG",linewidth=2.5)

    plt.legend(loc='lower right',prop={'size': 12})
    plt.title("Diurnal Cycle of Precipitation",fontweight='bold')
    ax.set_ylabel("Total Precipitation (mm/hr)",fontweight='bold')
    plt.xlabel("Local Time",fontweight='bold')

#plt.xtickslabels(xmarks)

    fig.tight_layout()

#cdo.cleanTempDir()
    plt.savefig(plot_name)
    fig.savefig("Fig7_Line_Diurnal.pdf", bbox_inches='tight', pad_inches=0.1)
