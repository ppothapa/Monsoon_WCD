import matplotlib.pyplot as plt
import numpy as np

# Actual region names
regions = ['South Asia', 'North Africa', 'South Africa', 'North America', 'South America', 'Australia']

# Replace the mock data below with your actual score dictionaries
# Example structure; fill in your real values
scores_08 = {
    'South Asia': {'Hit Rate': 0.8548, 'False Alarm Rate': 0.2161,'Bias Score': 1.0904, 'Accuracy': 0.9025, 'CSI': 0.6918} ,
    'North Africa': { 'Hit Rate': 0.8958,'False Alarm Rate': 0.3614,'Bias Score': 1.4028,'Accuracy': 0.8872,'CSI': 0.5945} ,
    'South Africa': {'Hit Rate': 0.9350, 'False Alarm Rate': 0.2890, 'Bias Score': 1.3150, 'Accuracy': 0.9058, 'CSI': 0.6775} ,
    'North America': {'Hit Rate': 0.8119, 'False Alarm Rate': 0.1881, 'Bias Score': 1.0000, 'Accuracy': 0.8967, 'CSI': 0.6833} ,
    'South America': {'Hit Rate': 0.8657, 'False Alarm Rate': 0.2418, 'Bias Score': 1.1418, 'Accuracy': 0.9229, 'CSI': 0.6784} ,
    'Australia': {'Hit Rate': 0.8750, 'False Alarm Rate': 0.3121, 'Bias Score': 1.2721, 'Accuracy': 0.9249, 'CSI': 0.6263} 

}


scores_06 = {
    'South Asia': {'Hit Rate': 0.8658,'False Alarm Rate': 0.2311,'Bias Score': 1.1260,'Accuracy': 0.8990,'CSI': 0.6870} ,
    'North Africa': {'Hit Rate': 0.8958,'False Alarm Rate': 0.3418,'Bias Score': 1.3611,'Accuracy': 0.8949,'CSI': 0.6114} ,
    'South Africa': {'Hit Rate': 0.9300,'False Alarm Rate': 0.3086,'Bias Score': 1.3450,'Accuracy': 0.8974, 'CSI': 0.6572} ,
    'North America': {'Hit Rate': 0.8911, 'False Alarm Rate': 0.2241, 'Bias Score': 1.1485, 'Accuracy': 0.8995, 'CSI': 0.7087} ,
    'South America': {'Hit Rate': 0.9776, 'False Alarm Rate': 0.2640, 'Bias Score': 1.3284, 'Accuracy': 0.9299, 'CSI': 0.7238} ,
    'Australia': {'Hit Rate': 0.9265, 'False Alarm Rate': 0.3000, 'Bias Score': 1.3235, 'Accuracy': 0.9323, 'CSI': 0.6632}
}

scores_05 = {
    'South Asia': {'Hit Rate': 0.8356, 'False Alarm Rate': 0.1931, 'Bias Score': 1.0356, 'Accuracy': 0.9067, 'CSI': 0.6963} ,
    'North Africa': {'Hit Rate': 0.9028, 'False Alarm Rate': 0.3048, 'Bias Score': 1.2986, 'Accuracy': 0.9090, 'CSI': 0.6468} ,
    'South Africa': {'Hit Rate': 0.8700, 'False Alarm Rate': 0.2898, 'Bias Score': 1.2250, 'Accuracy': 0.8974, 'CSI': 0.6421} ,
    'North America': {'Hit Rate': 0.8812, 'False Alarm Rate': 0.2261,'Bias Score': 1.1386, 'Accuracy': 0.8967, 'CSI': 0.7008} ,
    'South America': {'Hit Rate': 0.9179, 'False Alarm Rate': 0.2679, 'Bias Score': 1.2537,'Accuracy': 0.9215, 'CSI': 0.6872} ,
    'Australia': {'Hit Rate': 0.9485, 'False Alarm Rate': 0.2500, 'Bias Score': 1.2647, 'Accuracy': 0.9471, 'CSI': 0.7207}
}

# Settings
score_names = ['Hit Rate', 'False Alarm Rate', 'Bias Score', 'Accuracy', 'CSI']
model_names = ['ICON(10KM)', 'ICON(40KM)', 'ICON(80KM)']
COLORS      = ['r','g','b']
all_scores = [scores_08, scores_06, scores_05]
bar_width = 0.25
x = np.arange(len(regions))

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

# Loop over each score type
for idx, score_name in enumerate(score_names):
    ax = axes[idx]
    for i, score_dict in enumerate(all_scores):
        values = [score_dict[region][score_name] for region in regions]
        ax.bar(x + i * bar_width, values, width=bar_width, label=model_names[i],color=COLORS[i])

    ax.set_title(score_name, fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    if idx == 1:
        ax.legend()
    if idx == 2:
	       ax.set_ylim(0, 1.6)

# Remove empty subplot if unused
if len(score_names) % 2 != 0:
    fig.delaxes(axes[-1])

fig.suptitle("Skill Scores of ICON compared against MSWEP", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('all_skills')

