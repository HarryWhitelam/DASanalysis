import matplotlib.pyplot as plt
import pandas as pd

fname = 'wh'
target_chas = [750, 788, 875, 1475]
fs = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

df_spearman = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_corr_spearman.csv', index_col=(0,1))
df_std = pd.read_csv(f'/data/localraid/weather_corrs/{fname}_std_corr_spearman.csv', index_col=(0,1))

df_spearman.columns = pd.to_datetime(df_spearman.columns).strftime('%Y-%m')
df_std.columns = pd.to_datetime(df_std.columns).strftime('%Y-%m')

fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)
for ax, cha in zip(axs.flat, target_chas):
    df_spearman_cha = df_spearman.loc[cha].T
    df_spearman_cha = df_spearman_cha[fs]
    df_std_cha = df_std.loc[cha].T
    df_std_cha = df_std_cha[fs]
    df_spearman_cha.plot(ax=ax, marker='D', cmap='viridis', yerr=df_std_cha)
    ax.set_title(f'Channel {cha}', fontsize=14)
    ax.set_ylabel('Spearman coefficient', fontsize=14)
    ax.tick_params(labelrotation=90)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(axis='y')
    ax.legend().set_visible(False)
    ax.set_ylim(-0.19, 0.91)
    ax.axvline(6, color=(1, 0, 0, 0.4))
    # ax.axvline(9, color=(1, 0, 0, 0.4))
    # ax.axvline(10, color=(1, 0, 0, 0.4))
    ax.axvline(14, color=(1, 0, 0, 0.4))
    ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], minor=True)
axs.flat[-1].legend(loc='center left', bbox_to_anchor=(1,1), title='f (Hz)').set_visible(True)
fig.suptitle('DAS correlation with significant wave height')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.130)
plt.show()
