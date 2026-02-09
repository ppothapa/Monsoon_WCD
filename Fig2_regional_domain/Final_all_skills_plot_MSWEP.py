import matplotlib.pyplot as plt
import numpy as np

# Actual region names
regions = ['South Asia', 'North Africa', 'South Africa', 'North America', 'South America', 'Australia']

Titles = ['SAsiaM/EAsiaM', 'WAfriM', 'SAfriM', 'NAmerM', 'SAmerM','AusMCM']

# Replace the mock data below with your actual score dictionaries
# Example structure; fill in your real values

scores_08 = {
    'South Asia': {'Hit Rate': 0.8385, 'False Alarm Rate': 0.1769, 'Bias Score': 1.0187, 'Accuracy': 0.8858},
    'North Africa': {'Hit Rate': 0.8787, 'False Alarm Rate': 0.2969, 'Bias Score': 1.2497, 'Accuracy': 0.8922},
    'South Africa': {'Hit Rate': 0.8708, 'False Alarm Rate': 0.2886, 'Bias Score': 1.2240, 'Accuracy': 0.8848},
    'North America': {'Hit Rate': 0.8547, 'False Alarm Rate': 0.2018, 'Bias Score': 1.0707, 'Accuracy': 0.8971},
    'South America': {'Hit Rate': 0.9230, 'False Alarm Rate': 0.2272, 'Bias Score': 1.1944, 'Accuracy': 0.9304},
    'Australia': {'Hit Rate': 0.9330, 'False Alarm Rate': 0.2667, 'Bias Score': 1.2724, 'Accuracy': 0.9340}
}


scores_06 = {
    'South Asia': {'Hit Rate': 0.8464, 'False Alarm Rate': 0.1850, 'Bias Score': 1.0385, 'Accuracy': 0.8844},
    'North Africa': {'Hit Rate': 0.8693, 'False Alarm Rate': 0.3274, 'Bias Score': 1.2925, 'Accuracy': 0.8788},
    'South Africa': {'Hit Rate': 0.9065, 'False Alarm Rate': 0.2891, 'Bias Score': 1.2752, 'Accuracy': 0.8897},
    'North America': {'Hit Rate': 0.8507, 'False Alarm Rate': 0.2067, 'Bias Score': 1.0723, 'Accuracy': 0.8944},
    'South America': {'Hit Rate': 0.9368, 'False Alarm Rate': 0.2348, 'Bias Score': 1.2243, 'Accuracy': 0.9299},
    'Australia': {'Hit Rate': 0.9401, 'False Alarm Rate': 0.2893, 'Bias Score': 1.3228, 'Accuracy': 0.9281}
}


scores_05 = {
    'South Asia': {'Hit Rate': 0.8360, 'False Alarm Rate': 0.1910, 'Bias Score': 1.0334, 'Accuracy': 0.8792},
    'North Africa': {'Hit Rate': 0.8805, 'False Alarm Rate': 0.3494, 'Bias Score': 1.3532, 'Accuracy': 0.8704},
    'South Africa': {'Hit Rate': 0.9019, 'False Alarm Rate': 0.2836, 'Bias Score': 1.2590, 'Accuracy': 0.8913},
    'North America': {'Hit Rate': 0.8053, 'False Alarm Rate': 0.1435, 'Bias Score': 0.9402, 'Accuracy': 0.9062},
    'South America': {'Hit Rate': 0.8777, 'False Alarm Rate': 0.2118, 'Bias Score': 1.1135, 'Accuracy': 0.9284},
    'Australia': {'Hit Rate': 0.9055, 'False Alarm Rate': 0.3155, 'Bias Score': 1.3228, 'Accuracy': 0.9169}
}


# Settings
score_names = ['Hit Rate', 'False Alarm Rate', 'Bias Score', 'Accuracy']
score_names_title = ['(a) Hit Rate', '(b) False Alarm Rate', '(c) Bias Score', '(d) Accuracy']
model_names = ['ICON(10KM)', 'ICON(40KM)', 'ICON(80KM)']
COLORS      = ['r','g','b']
COLORS = ['darkblue', 'royalblue', 'lightsteelblue']  # Renamed from 'colors'

all_scores = [scores_08, scores_06, scores_05]
bar_width = 0.25
x = np.arange(len(regions))

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Loop over each score type
for idx, score_name in enumerate(score_names):
    ax = axes[idx]
    for i, score_dict in enumerate(all_scores):
        values = [score_dict[region][score_name] for region in regions]
        ax.bar(x + i * bar_width, values, width=bar_width, label=model_names[i],color=COLORS[i])
    
    ax.set_title(score_names_title[idx], fontsize=20,fontweight='bold')

    print(ax)
    print(idx)
    if idx== 0:
       ax.set_xticks([])
       ax.set_xticklabels([])  # Remove tick label
   #ax.set_xticklabels(regions, rotation=45, ha='right',fontweight='bold', fontsize=12)
    elif idx== 1:
       ax.set_xticks([])
       ax.set_xticklabels([])  # Remove tick label
       #ax.set_xticklabels(regions, rotation=45, ha='right',fontweight='bold', fontsize=12)
    else:
       ax.set_xticks(x + bar_width)
       ax.set_xticklabels(Titles, rotation=45, ha='right',fontweight='bold', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontsize=14)
    plt.setp(ax.get_yticklabels(), fontweight='bold',fontsize=14)
    if idx == 1:
        ax.legend(prop={'weight': 'bold', 'size': 20})
    if idx == 2:
               ax.set_ylim(0, 1.6)


# Remove empty subplot if unused
if len(score_names) % 2 != 0:
    fig.delaxes(axes[-1])

fig.suptitle("Skill Scores of ICON compared against MSWEP", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('all_skills_MSWEP')
fig.savefig("all_skills_MSWEP.pdf", bbox_inches='tight', pad_inches=0.1)

