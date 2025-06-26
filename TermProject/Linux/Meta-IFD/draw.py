import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('result/Meta-IFD_Phish.csv')

# Add index column
df.insert(0, 'Index', range(1,len(df)+1))

# Figure setup
fig, ax = plt.subplots(figsize=(18, 3.5))
ax.axis('off')

# Render the table with bold headers
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0,0,1,1]
)

# Bold header labels and adjust style
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')  # Bold header
        cell.set_fontsize(13)
        cell.set_facecolor('#dddddd')
    else:
        cell.set_fontsize(12)

table.auto_set_column_width(col=list(range(len(df.columns))))

plt.savefig('result/Table.png', bbox_inches='tight', dpi=300)
plt.close()
