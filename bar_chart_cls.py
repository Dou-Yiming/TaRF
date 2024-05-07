import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

chance = [18.6, 66.1, 56.3]
of = [36.2, 72.0, 69.0]
vg = [39.1, 69.4, 70.4]
tg = [54.7, 77.3, 79.4]
tgof = [54.6, 87.3, 84.8]
tgvg = [53.1, 86.7, 83.6]
tgr = [57.6, 88.4, 81.7]
tgre = [59.0, 88.7, 86.1]

# Create a DataFrame
data = {
    # 'Chance': chance,
    # 'ObjectFolder 2.0': of,
    # 'VisGel': vg,
    'TG': tg,
    'TG + OF 2.0': tgof,
    'TG + VisGel': tgvg,
    'TG + TaRF (Real)': tgr,
    'TG + TaRF (Real + Estimated)': tgre,
}
categories = ['Material', 'Hard/Soft', 'Rough/Smooth']
df = pd.DataFrame(data, index=categories)

# Melt the DataFrame
df_melted = df.reset_index().melt(id_vars='index')

# Create the bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='index', y='value', hue='variable', data=df_melted, palette='Spectral', saturation=0.8, dodge=True)

# Rename the axes
plt.xlabel('')
plt.ylabel('')

# Set the ylim if needed
plt.ylim(50, 100)

# Set the legend title
plt.legend(title='',loc="upper left",bbox_to_anchor=[0,1], 
            ncol = 2,frameon=False, fontsize=14)

plt.xticks(rotation=0, fontsize=14)
plt.yticks(rotation=0, fontsize=14)
# plt.title('Cross-modal Retrieval Results.', fontsize=18)

# Adding the text labels
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.1f' % float(p.get_height()),
            fontsize=12, ha='center', va='bottom')

# Show the plot
plt.margins(0.03, tight=True)
plt.savefig('assets/figs/classification_results.jpg', dpi=200, bbox_inches='tight')
