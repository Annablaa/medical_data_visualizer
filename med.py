import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')
def calculate_overweight(df):
    if df['cholesterol'] == 1 or df['gluc'] == 1:
        return 0
    elif df['cholesterol'] > 1 or df['gluc'] > 1:
        return 1
    else:
        return None

# Apply the custom function to create the 'overweight' column
df['overweight'] = df.apply(calculate_overweight, axis=1)
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df,value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = pd.melt(df,id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    feature_counts = df_cat[df_cat['value'].isin([0, 1])].groupby(['variable', 'cardio', 'value']).size().reset_index(name='count')
    sns.set(style='whitegrid')
    g = sns.catplot(
        data=feature_counts,
        x='variable',
        y='count',
        hue='value',
        col='cardio',
        kind='bar',
        height=6,
        aspect=1, 
        col_order=[0, 1]  
    )

    # Customize the plot
    g.set_axis_labels('Feature', 'Total')
    g.set_titles("Cardio = {col_name} ")
    g.despine(left=True)
    plt.tight_layout()
    # Show the plot
    plt.show()

draw_cat_plot()

def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(
        corr,
        annot=True,
        fmt=".1f",
        linewidths=.5,
        mask=mask,
        square=True,
        vmin=-0.12,
        vmax=0.28,
        center=0.0,
        ax = ax,
        cbar_kws={'ticks': np.arange(-0.08,0.25,0.08),'use_gridspec': False}
    )
    
    
    # Save the figure
    fig.savefig('heatmap.png')

    return fig

# Call the function to draw the heatmap and get the figure
resulting_fig = draw_heat_map()

# Display the figure
plt.show()
