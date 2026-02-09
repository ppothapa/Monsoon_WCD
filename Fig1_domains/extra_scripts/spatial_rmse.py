import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['IMERG', 'MSWEP', 'ERA5', 'CMORPH', 'CPC', 'GPCC']
icon08_rmse = [2.679, 2.843, 2.720, 3.172, 2.751, 2.524]
icon06_rmse = [2.348, 2.659, 2.582, 2.918, 2.513, 2.381] # Your best performer
icon05_rmse = [2.341, 2.622, 2.594, 2.750, 2.489, 2.552]

x = np.arange(len(datasets))
width = 0.25

# Create figure with higher DPI for publication
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Create bars with your chosen color scheme
rects1 = ax.bar(x - width, icon08_rmse, width, label='ICON-08', color='darkblue')
rects2 = ax.bar(x, icon06_rmse, width, label='ICON-06', color='royalblue')
rects3 = ax.bar(x + width, icon05_rmse, width, label='ICON-05', color='lightsteelblue')

# Set labels with BOLD fonts and increased size
ax.set_ylabel('RMSE (mm day$^{-1}$)', fontweight='bold', fontsize=14)
ax.set_xlabel('Reference Dataset', fontweight='bold', fontsize=14)
ax.set_title('Root-Mean-Square Error (RMSE) of Precipitation by Model Resolution', 
             fontweight='bold', fontsize=16)

# Set ticks and make labels BOLD
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontweight='bold', fontsize=12)
ax.set_yticks(np.arange(2.0, 3.6, 0.2))  # Custom y-ticks for better readability

# Make legend BOLD
ax.legend(prop={'weight': 'bold', 'size': 12})

# Set y-axis limit to focus on the differences
ax.set_ylim(2.0, 3.5)

# Make tick parameters bold and larger
ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
ax.tick_params(axis='both', which='minor', labelsize=10, width=1, length=4)

# Add horizontal grid lines for easier reading (behind the bars)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)

# Make spines (axis lines) bolder
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Rotate x-axis labels for better fit
plt.xticks(rotation=45, ha='right')  # 'ha' is horizontal alignment

plt.tight_layout()
plt.savefig('rmse_analysis.png', bbox_inches='tight', dpi=300)  # Save high-resolution version
