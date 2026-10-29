import pandas as pd
import matplotlib.pyplot as plt

def plot_coherence(cha):
    if cha == 'nodes': 
        df = pd.read_csv(f'../Nodes_Analysis/checkpoints/node_coherence.csv', index_col=0)
    else:
        df = pd.read_csv(f'../Nodes_Analysis/checkpoints/das_coherence_{cha}.csv', index_col=0)
    df = df.drop(columns=['0.02'])
    df = df.rename(columns={'0.07142857142857142':'0.07 Hz', '0.1': '0.1 Hz', '1.0': '1.0 Hz', '1.5': '1.5 Hz', '5.0': '5.0 Hz', '10.0':'10.0 Hz'})
    ax = df.plot(cmap='viridis')
    ax.set_xlim(0,6)
    ax.set_ylim(0,0.8)
    ax.axvline(2.0, c='k', ls='--', alpha=0.5)
    ax.text(2.0, 0.99, 'S2', ha='left', va='top', rotation=90, transform=ax.get_xaxis_transform())
    ax.axvline(1.93, c='k', ls='--', alpha=0.5)
    ax.text(1.93, 0.99, 'M2', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())
    ax.axvline(3.86, c='k', ls='--', alpha=0.5)
    ax.text(3.86, 0.99, 'M4', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())
    ax.set_xlabel('Cycles/day')
    ax.set_ylabel('Magnitude-squared coherence')
    plt.tight_layout()
    plt.show()

