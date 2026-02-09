import matplotlib.pyplot as plt
import numpy as np

# Example mockup data
regions = ['South Asia', 'North Africa' , 'South Africa' , 'North America' , 'South America' , 'Australia']
score_name = 'Hit Rate'  # or any of the scores

scores_08 = {'South Asia': {'Hit Rate': 0.8548 }, 'North Africa': {'Hit Rate': 0.8958}, 'South Africa': {'Hit Rate': 0.9350}, 'North America': {'Hit Rate': 0.8119},  'South America':{'Hit Rate': 0.8657},  'Australia':{'Hit Rate': 0.8750} }
scores_06 = {'South Asia': {'Hit Rate': 0.8658 }, 'North Africa': {'Hit Rate': 0.8958}, 'South Africa': {'Hit Rate': 0.9300}, 'North America': {'Hit Rate': 0.8911},   'South America':{'Hit Rate': 0.9776},  'Australia':{'Hit Rate': 0.9265} }
scores_05 = {'South Asia': {'Hit Rate': 0.8356}, 'North Africa': {'Hit Rate': 0.9028}, 'South Africa': {'Hit Rate': 0.8700}, 'North America': {'Hit Rate': 0.8812},   'South America':{'Hit Rate':  0.9179}, 'Australia':{'Hit Rate': 0.9485} }



# Collect data for plotting
model_names = ['ICON (10KM)', 'ICON (40KM)', 'ICON (80KM)']
all_scores = [scores_08, scores_06, scores_05]
bar_width = 0.25
x = np.arange(len(regions))

plt.figure(figsize=(10, 6))

for i, (model, score_dict) in enumerate(zip(model_names, all_scores)):
    values = [score_dict[region][score_name] for region in regions]
    plt.bar(x + i*bar_width, values, width=bar_width, label=model)

plt.xticks(x + bar_width, regions)
plt.ylabel(score_name)
plt.title(f'{score_name} across Regions and Models')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Skills')
