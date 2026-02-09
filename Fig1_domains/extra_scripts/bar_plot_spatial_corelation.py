import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['IMERG', 'MSWEP', 'ERA5', 'CMORPH', 'GPCC', 'CPC']
icon08 = [0.765, 0.745, 0.754, 0.715, 0.759, 0.647]
icon06 = [0.805, 0.758, 0.758, 0.740, 0.778, 0.659]
icon05 = [0.777, 0.723, 0.720, 0.710, 0.742, 0.592]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, icon08, width, label='ICON (10 km)',color='darkblue' )
rects2 = ax.bar(x, icon06, width, label='ICON (40 km)',color='royalblue')
rects3 = ax.bar(x + width, icon05, width, label='ICON (80 km)',color='lightsteelblue')


# Set up the plot with bold fonts for publication
ax.set_ylabel('Spatial Correlation (r)', fontweight='bold', fontsize=12)
ax.set_xlabel('Reference Dataset', fontweight='bold', fontsize=12) # Add X-axis label if appropriate
ax.set_title('Spatial Correlation of Precipitation Patterns by Model Resolution and Reference Dataset', 
             fontweight='bold', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontweight='bold')  # Make dataset labels bold too

# Make legend bold
ax.legend(prop={'weight': 'bold'})

ax.set_ylim(0.5, 0.85) # Set y-axis to focus on the differences

# Make tick labels bold as well
ax.tick_params(axis='both', which='major', labelsize=10, width=2)  # Increased font size and added width
ax.tick_params(axis='both', which='minor', labelsize=8, width=1)



plt.xticks(rotation=45)
plt.tight_layout()

plot_name='spatial_correlation'
plt.savefig(plot_name,dpi=600, bbox_inches='tight', pad_inches=0.1)
fig.savefig("Spatial_Correlation_GM.pdf", bbox_inches='tight', pad_inches=0.1)
