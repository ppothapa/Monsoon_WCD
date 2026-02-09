import matplotlib.pyplot as plt
import numpy as np

# Actual region names
regions = ['South Asia', 'North Africa', 'South Africa', 'North America', 'South America', 'Australia']

# Replace the mock data below with your actual score dictionaries
# Example structure; fill in your real values

scores_08 = {
    'South Asia': {'Hit Rate': 0.7844, 'False Alarm Rate': 0.1245, 'Bias Score': 0.8960, 'Accuracy': 0.8757, 'CSI': 0.7057},
    'North Africa': {'Hit Rate': 0.8361, 'False Alarm Rate': 0.2248, 'Bias Score': 1.0786, 'Accuracy': 0.8969, 'CSI': 0.6729},
    'South Africa': {'Hit Rate': 0.8522, 'False Alarm Rate': 0.1367, 'Bias Score': 0.9871, 'Accuracy': 0.9163, 'CSI': 0.7509},
    'North America': {'Hit Rate': 0.8071, 'False Alarm Rate': 0.1651, 'Bias Score': 0.9666, 'Accuracy': 0.8888, 'CSI': 0.6960},
    'South America': {'Hit Rate': 0.9120, 'False Alarm Rate': 0.1837, 'Bias Score': 1.1172, 'Accuracy': 0.9373, 'CSI': 0.7566},
    'Australia': {'Hit Rate': 0.9019, 'False Alarm Rate': 0.2057, 'Bias Score': 1.1355, 'Accuracy': 0.9396, 'CSI': 0.7311}
}


scores_06 = {
    'South Asia': {'Hit Rate': 0.7871, 'False Alarm Rate': 0.1383, 'Bias Score': 0.9134, 'Accuracy': 0.8711, 'CSI': 0.6988},
    'North Africa': {'Hit Rate': 0.8357, 'False Alarm Rate': 0.2508, 'Bias Score': 1.1155, 'Accuracy': 0.8874, 'CSI': 0.6530},
    'South Africa': {'Hit Rate': 0.8845, 'False Alarm Rate': 0.1399, 'Bias Score': 1.0284, 'Accuracy': 0.9232, 'CSI': 0.7733},
    'North America': {'Hit Rate': 0.8012, 'False Alarm Rate': 0.1724, 'Bias Score': 0.9681, 'Accuracy': 0.8846, 'CSI': 0.6866},
    'South America': {'Hit Rate': 0.9351, 'False Alarm Rate': 0.1835, 'Bias Score': 1.1452, 'Accuracy': 0.9413, 'CSI': 0.7727},
    'Australia': {'Hit Rate': 0.9114, 'False Alarm Rate': 0.2280, 'Bias Score': 1.1805, 'Accuracy': 0.9349, 'CSI': 0.7181}
}


scores_05 = {
    'South Asia': {'Hit Rate': 0.7859, 'False Alarm Rate': 0.1354, 'Bias Score': 0.9089, 'Accuracy': 0.8719, 'CSI': 0.6998},
    'North Africa': {'Hit Rate': 0.8426, 'False Alarm Rate': 0.2785, 'Bias Score': 1.1679, 'Accuracy': 0.8776, 'CSI': 0.6358},
    'South Africa': {'Hit Rate': 0.8499, 'False Alarm Rate': 0.1630, 'Bias Score': 1.0154, 'Accuracy': 0.9066, 'CSI': 0.7293},
    'North America': {'Hit Rate': 0.7510, 'False Alarm Rate': 0.1153, 'Bias Score': 0.8488, 'Accuracy': 0.8906, 'CSI': 0.6841},
    'South America': {'Hit Rate': 0.8610, 'False Alarm Rate': 0.1734, 'Bias Score': 1.0416, 'Accuracy': 0.9317, 'CSI': 0.7293},
    'Australia': {'Hit Rate': 0.8737, 'False Alarm Rate': 0.2599, 'Bias Score': 1.1805, 'Accuracy': 0.9212, 'CSI': 0.6685}
}


# Settings
score_names = ['Hit Rate', 'False Alarm Rate', 'Bias Score', 'Accuracy', 'CSI']
model_names = ['ICON(10KM)', 'ICON(40KM)', 'ICON(80KM)']
COLORS      = ['r','g','b']
COLORS = ['darkblue', 'royalblue', 'lightsteelblue']  # Renamed from 'colors'

all_scores = [scores_08, scores_06, scores_05]
bar_width = 0.25
x = np.arange(len(regions))

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(16,8))
axes = axes.flatten()

# Loop over each score type
for idx, score_name in enumerate(score_names):
    ax = axes[idx]
    for i, score_dict in enumerate(all_scores):
        values = [score_dict[region][score_name] for region in regions]
        ax.bar(x + i * bar_width, values, width=bar_width, label=model_names[i],color=COLORS[i])

    ax.set_title(score_name, fontsize=14,fontweight='bold')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(regions, rotation=45, ha='right',fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    if idx == 1:
        ax.legend(prop={'weight': 'bold'})
    if idx == 2:
	       ax.set_ylim(0, 1.6)

# Remove empty subplot if unused
if len(score_names) % 2 != 0:
    fig.delaxes(axes[-1])

fig.suptitle("Skill Scores of ICON compared against IMERG", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('all_skills_IMERG')
fig.savefig("all_skills_IMERG.pdf", bbox_inches='tight', pad_inches=0.1)

