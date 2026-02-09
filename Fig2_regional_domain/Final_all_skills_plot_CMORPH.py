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
    'South Asia': {'Hit Rate': 0.7906, 'False Alarm Rate': 0.1841, 'Bias Score': 0.9690, 'Accuracy': 0.8611},
    'North Africa': {'Hit Rate': 0.8554, 'False Alarm Rate': 0.3176, 'Bias Score': 1.2534, 'Accuracy': 0.8775},
    'South Africa': {'Hit Rate': 0.8881, 'False Alarm Rate': 0.1644, 'Bias Score': 1.0629, 'Accuracy': 0.9179},
    'North America': {'Hit Rate': 0.8226, 'False Alarm Rate': 0.1442, 'Bias Score': 0.9612, 'Accuracy': 0.8996},
    'South America': {'Hit Rate': 0.9259, 'False Alarm Rate': 0.1970, 'Bias Score': 1.1531, 'Accuracy': 0.9361},
    'Australia': {'Hit Rate': 0.8930, 'False Alarm Rate': 0.2336, 'Bias Score': 1.1652, 'Accuracy': 0.9301}
}


scores_05 = {
    'South Asia': {'Hit Rate': 0.7841, 'False Alarm Rate': 0.1869, 'Bias Score': 0.9643, 'Accuracy': 0.8581},
    'North Africa': {'Hit Rate': 0.8589, 'False Alarm Rate': 0.3455, 'Bias Score': 1.3123, 'Accuracy': 0.8658},
    'South Africa': {'Hit Rate': 0.8671, 'False Alarm Rate': 0.1737, 'Bias Score': 1.0494, 'Accuracy': 0.9097},
    'North America': {'Hit Rate': 0.7638, 'False Alarm Rate': 0.0937, 'Bias Score': 0.8428, 'Accuracy': 0.8999},
    'South America': {'Hit Rate': 0.8574, 'False Alarm Rate': 0.1825, 'Bias Score': 1.0488, 'Accuracy': 0.9292},
    'Australia': {'Hit Rate': 0.8530, 'False Alarm Rate': 0.2680, 'Bias Score': 1.1652, 'Accuracy': 0.9153}
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

fig.suptitle("Skill Scores of ICON compared against CMORPH", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('all_skills_CMORPH')
fig.savefig("all_skills_CMORPH.pdf", bbox_inches='tight', pad_inches=0.1)

