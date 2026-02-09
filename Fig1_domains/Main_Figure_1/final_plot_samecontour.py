# create_master_plot_full.py
# Quick plotting script - NO CALCULATIONS, uses pre-computed values for ALL datasets
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib import colors as mcolors  # Rename the import to avoid conflict

# ---------------------------------------------------------------------------------
# 1. Load the saved data
# ---------------------------------------------------------------------------------
ds = xr.open_dataset('calculated_data_for_plotting_samecontour_40to40.nc')

# Extract variables for cleaner code
var_08_mag_na = ds['var_08_summer_winter_mag_na']
var_08_mag = ds['var_08_summer_winter_mag']
var_08_nh = ds['var_08_summer_winter_NH']
var_08_sh = ds['var_08_summer_winter_SH']

var_06_mag_na = ds['var_06_summer_winter_mag_na']
var_06_mag = ds['var_06_summer_winter_mag']
var_06_nh = ds['var_06_summer_winter_NH']
var_06_sh = ds['var_06_summer_winter_SH']

var_05_mag_na = ds['var_05_summer_winter_mag_na']
var_05_mag = ds['var_05_summer_winter_mag']
var_05_nh = ds['var_05_summer_winter_NH']
var_05_sh = ds['var_05_summer_winter_SH']


var_imerg_mag_na = ds['var_imerg_summer_winter_mag_na']
var_imerg_mag = ds['var_imerg_summer_winter_mag']
var_imerg_nh = ds['var_imerg_summer_winter_NH']
var_imerg_sh = ds['var_imerg_summer_winter_SH']

# ---------------------------------------------------------------------------------
# 2. Pre-computed metric values for ALL OBSERVATIONAL DATASETS
# ---------------------------------------------------------------------------------
# Spatial Correlation Values
corr_data = {
    'IMERG': [0.765, 0.805, 0.777],  # [ICON-08, ICON-06, ICON-05]
    'MSWEP': [0.745, 0.758, 0.723],
    'ERA5':  [0.754, 0.758, 0.720],
    'CMORPH':[0.715, 0.740, 0.710],
    'CPC':   [0.647, 0.659, 0.592],
    'GPCC':  [0.759, 0.778, 0.742]
}

# RMSE Values
rmse_data = {
    'IMERG': [2.679, 2.348, 2.341],
    'MSWEP': [2.843, 2.659, 2.622],
    'ERA5':  [2.720, 2.582, 2.594],
    'CMORPH':[3.172, 2.918, 2.750],
    'CPC':   [2.751, 2.513, 2.489],
    'GPCC':  [2.524, 2.381, 2.552]
}

datasets = list(corr_data.keys())
models = ['ICON (10 km)', 'ICON (40 km)', 'ICON (80 km)']
bar_colors = ['darkblue', 'royalblue', 'lightsteelblue']  # Renamed from 'colors'

# ---------------------------------------------------------------------------------
# 3. Define a helper function to create a spatial plot
# ---------------------------------------------------------------------------------
def create_spatial_plot(ax, data_mag, data_nh, data_sh, title):
    """Creates a standardized spatial plot on a given axis."""
    # Use a common max value for consistent coloring

    bounds = np.arange(0,16.1,1) # Here the limit is till 55 to 100, it is strange in python
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # Plot the magnitude
    plot = data_mag.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap='Spectral_r',
                                  norm=norm, add_colorbar=False, add_labels=False)
    
    # Add the NH and SH contours
    data_nh.plot.contour(ax=ax, levels=[2.0], transform=ccrs.PlateCarree(), 
                         colors=['k'], linewidths=0.5)
    data_sh.plot.contour(ax=ax, levels=[-2.0], transform=ccrs.PlateCarree(), 
                         colors=['k'], linewidths=0.5)
    
    # Add coastlines and title
    ax.coastlines(resolution='50m', linewidth=0.5, color='gray')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('auto')
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color="none", 
                      linewidth=1, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}
    
    return plot

# ---------------------------------------------------------------------------------
# 4. Create the master figure
# ---------------------------------------------------------------------------------
# Create a larger figure to accommodate all datasets
fig = plt.figure(figsize=(16, 12))
#gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.3)

gs = fig.add_gridspec(3, 2,height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
# Define projection
proj = ccrs.PlateCarree()

# Row 1: ICON Models
ax1 = fig.add_subplot(gs[0, 1], projection=proj)
create_spatial_plot(ax1, var_08_mag_na, var_08_mag, var_08_mag, '(b) ICON (10 km)')

ax2 = fig.add_subplot(gs[1, 0], projection=proj)
create_spatial_plot(ax2, var_06_mag_na, var_06_mag, var_06_mag, '(c) ICON (40 km)')

ax3 = fig.add_subplot(gs[1, 1], projection=proj)
create_spatial_plot(ax3, var_05_mag_na, var_05_mag, var_05_mag, '(d) ICON (80 km)')

# Row 2: IMERG (centered)
ax4 = fig.add_subplot(gs[0, 0], projection=proj)
create_spatial_plot(ax4, var_imerg_mag_na, var_imerg_mag, var_imerg_mag, '(a) IMERG (Reference)')

# Hide the empty subplots in row 2
#ax_empty1 = fig.add_subplot(gs[1, 0], projection=proj)
#ax_empty1.set_visible(False)
#ax_empty2 = fig.add_subplot(gs[1, 2], projection=proj)
#ax_empty2.set_visible(False)

# Prepare data for grouped bar plot (DEFINE x HERE, before using it)
x = np.arange(len(datasets))  # the label locations
width = 0.25  # the width of the bars

# Row 3: Correlation bar plot (span full width)
ax5 = fig.add_subplot(gs[2, 0])  # Takes entire third row

# Create grouped bar plot for Correlation on ax5
for i, model in enumerate(models):
    corr_values = [corr_data[ds][i] for ds in datasets]
    ax5.bar(x + (i-1)*width, corr_values, width, label=model, color=bar_colors[i])

ax5.set_ylabel('Spatial Correlation (r)', fontweight='bold', fontsize=12)
ax5.set_title('(e) Spatial Correlation Across All Reference Datasets', fontweight='bold', fontsize=14)
ax5.set_xticks(x)
ax5.set_xticklabels(datasets, fontweight='bold', fontsize=12)
ax5.set_ylim(0.5, 0.85)
#ax5.legend(prop={'weight': 'bold'})
ax5.grid(axis='y', linestyle='--', alpha=0.7)

# Row 4: RMSE bar plot (span full width)
ax6 = fig.add_subplot(gs[2, 1])  # Takes entire fourth row

# Create grouped bar plot for RMSE on ax6
for i, model in enumerate(models):
    rmse_values = [rmse_data[ds][i] for ds in datasets]
    ax6.bar(x + (i-1)*width, rmse_values, width, label=model, color=bar_colors[i])

ax6.set_ylabel('RMSE (mm day$^{-1}$)', fontweight='bold', fontsize=12)
ax6.set_title('(f) RMSE Across All Reference Datasets', fontweight='bold', fontsize=14)
ax6.set_xticks(x)
ax6.set_xticklabels(datasets, fontweight='bold', fontsize=12)
ax6.set_ylim(2.0, 3.5)
ax6.legend(prop={'weight': 'bold', 'size': 14})
ax6.grid(axis='y', linestyle='--', alpha=0.7)

# ---------------------------------------------------------------------------------
# 5. Add a colorbar for the spatial plots
# ---------------------------------------------------------------------------------
# Add a colorbar to the right of the spatial plots
bounds = np.arange(0,16.1,1) # Here the limit is till 55 to 100, it is strange in python
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)
cax = fig.add_axes([0.92, 0.45, 0.02, 0.4])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap='Spectral_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax,extend='both')
cbar.set_label('Precipitation (mm day$^{-1}$)', fontweight='bold', fontsize=16)

## 6. Add subpanel names to the figures


# ---------------------------------------------------------------------------------
# 7. Save the figure
# ---------------------------------------------------------------------------------
plt.tight_layout()
plt.savefig('master_plot_comprehensive.png', dpi=300, bbox_inches='tight')
fig.savefig("Master_samecontour.pdf", bbox_inches='tight', pad_inches=0.1)
#plt.show()
