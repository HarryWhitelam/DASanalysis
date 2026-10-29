from pathlib import Path
import matplotlib.pyplot as plt
from obspy.signal import PPSD


def save_ppsd_plots(ppsd, title, outdir, prefix):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = ppsd.plot(
        xaxis_frequency=True,
        show_coverage=False,
        period_lim=(0.02, 50),
        show=False
    )

    ax = fig.axes[0]
    ax.set_ylim(-180, -90)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outdir / f"ppsd_{prefix}.eps")
    plt.close(fig)

    fig = ppsd.plot_spectrogram(
        clim=[-180, -100],
        cmap="jet",
        yaxis_frequency=True,
        show=False
    )

    fig.axes[0].set_title(title)

    plt.tight_layout()
    plt.savefig(outdir / f"spectrogram_{prefix}.eps")
    plt.close(fig)


def process_ppsd(stream, inventory, title, outdir, ppsd_length=900, period_limits=(0.02, 100)):
    ppsd = PPSD(
        stream[0].stats,
        metadata=inventory,
        ppsd_length=ppsd_length,
        period_limits=period_limits,
    )

    ppsd.add(stream)

    prefix = f"{stream[0].id}_{stream[0].stats.starttime}_{stream[-1].stats.endtime}"

    save_ppsd_plots(
        ppsd,
        title,
        outdir,
        prefix
    )
