import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_cli_args():
    """Parse command-line arguments for Holi clustering measurements plots."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--statistics', nargs='+', default=['spectrum'])
    parser.add_argument("--start_phase", type=int, default=201)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument('--tracer', type=str, default='LRG')
    parser.add_argument('--region', type=str, default='NGC')
    parser.add_argument('--zrange', nargs=2, type=float, default=[0.4, 0.6])
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/pscratch/sd/a/acasella/acm/dr2/measurements/holi'
    )

    args = parser.parse_args()
    return args


def plot_pspectrum(data_fn, directory, fn):
    from jaxpower import read

    data = read(data_fn)

    ells = data.ells
    poles = [data.get(ell) for ell in ells]
    k = poles[0].coords('k')
    poles = [pole.value() for pole in poles]

    fig, ax = plt.subplots(figsize=(10,7))

    for ell, pole in zip(ells, poles):
        ax.plot(k, k*pole, label=fr'$\ell={ell}$')

    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(r'$k\, [h\,{\rm Mpc}^{-1}]$', fontsize=15)
    ax.set_ylabel(r'$k P_\ell(k)\, [h^{-2}\,{\rm Mpc}]$', fontsize=15)

    outfile = Path(directory) / f"{fn}_acm.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {outfile}")


def plot_density_split(data_fn, directory, fn):
    data = np.load(data_fn, allow_pickle=True)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    for q, tpcf in enumerate(data):
        tpcf = tpcf[::4]
        s, multipoles = tpcf(ells=(0, 2), return_sep=True)

        axes[0].plot(
            s,
            s**2 * multipoles[0],
            color=colors[q % len(colors)],
            label=f'Q{q+1}'
        )

        axes[1].plot(
            s,
            s**2 * multipoles[1],
            color=colors[q % len(colors)]
        )

    axes[0].set_ylabel(r'$s^2\xi_0(s)$', fontsize=15)
    axes[1].set_ylabel(r'$s^2\xi_2(s)$', fontsize=15)
    axes[1].set_xlabel(r'$s\, [h^{-1}\mathrm{Mpc}]$', fontsize=15)

    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)

    axes[0].tick_params(axis='both', labelsize=14)
    axes[1].tick_params(axis='both', labelsize=14)

    outfile = Path(directory) / f"{fn}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {outfile}")


def plot_minkowski(data_fn, directory, fn):
    mf_cons = [1, 1e3, 1e5, 1e7]
    ylabels = [
        r"$V_{0}$",
        r"$V_{1}[10^{- "+str(int(np.log10(mf_cons[1])))+"}hMpc^{-1}]$",
        r"$V_{2}[10^{- "+str(int(np.log10(mf_cons[2])))+"}(hMpc^{-1})^2]$",
        r"$V_{3}[10^{- "+str(int(np.log10(mf_cons[3])))+"}(hMpc^{-1})^3]$"
    ]
    fig = plt.figure(constrained_layout=False, figsize=[10, 10])
    spec = fig.add_gridspec(ncols=2, nrows=2, hspace=0.2, wspace=0.3)
    axes = []
    for i in range(4):
        ii = i // 2
        jj = i % 2
        ax = fig.add_subplot(spec[ii, jj])
        ax.set_xlabel(r"$\delta$", fontsize=15)
        ax.set_ylabel(ylabels[i], fontsize=15)
        ax.axhline(0, color="black")
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True)
        axes.append(ax)

    MFs = np.load(data_fn)
    thresholds = np.linspace(-1, 5, MFs.shape[0], dtype=np.float32)
    for i in range(4):
        axes[i].plot(thresholds, MFs[:, i] * mf_cons[i], color="blue", linewidth=0.7)

    plt.tight_layout()
    outfile = Path(directory) / f"{fn}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {outfile}")


def plot_wst(data_fn, directory, fn):
    fig, ax = plt.subplots(figsize=(10, 7))

    coeffs = np.load(data_fn)
    x = np.arange(len(coeffs))

    ax.set_xlabel("WST coefficient order", fontsize=15)
    ax.set_ylabel("WST coefficient", fontsize=15)

    ax.plot(x, coeffs, color="blue")
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    outfile = Path(directory) / f"{fn}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {outfile}")
    

if __name__ == '__main__':
    args = get_cli_args()

    tracer = args.tracer
    region = args.region
    zmin, zmax = args.zrange
    base_dir=args.base_dir
    stats = args.statistics
    phases = list(range(args.start_phase, args.start_phase + args.n_phase))

    for stat in stats:
        if stat == "spectrum":
            data_dir = Path(base_dir) / stat
            fn = f'mesh2_poles_{tracer}_{region}_z{zmin}-{zmax}'

            for phase_idx in phases:
                directory = Path(data_dir) / f'ph{phase_idx}'
                data_fn = Path(directory) / f'{fn}_acm.h5'
                if not data_fn.exists():
                    print(f'Skipping phase {phase_idx}: missing data catalog {data_fn}')
                    continue

                plot_pspectrum(data_fn, directory, fn)

        if stat == "density_split":
            data_dir = Path(base_dir) / stat
            kinds = ['q', 'g']

            for kind in kinds:
                fn = f'dsc_xiq{kind}_poles_{tracer}_{region}_z{zmin}-{zmax}'

                for phase_idx in phases:
                    directory = Path(data_dir) / f'ph{phase_idx}'
                    data_fn = Path(directory) / f'{fn}.npy'
                    if not data_fn.exists():
                        print(f'Skipping phase {phase_idx}: missing data catalog {data_fn}')
                        continue

                    plot_density_split(data_fn, directory, fn)

        if stat == "minkowski":
            data_dir = Path(base_dir) / stat
            fn = f'MFs_{tracer}_{region}_z{zmin}-{zmax}'

            for phase_idx in phases:
                directory = Path(data_dir) / f'ph{phase_idx}'
                data_fn = Path(directory) / f'{fn}.npy'
                if not data_fn.exists():
                    print(f'Skipping phase {phase_idx}: missing data catalog {data_fn}')
                    continue
                plot_minkowski(data_fn, directory, fn)


        if stat == "wst":
            data_dir = Path(base_dir) / stat
            fn = f'wst_{tracer}_{region}_z{zmin}-{zmax}_jax'

            for phase_idx in phases:
                directory = Path(data_dir) / f'ph{phase_idx}'
                data_fn = Path(directory) / f'{fn}.npy'
                if not data_fn.exists():
                    print(f'Skipping phase {phase_idx}: missing data catalog {data_fn}')
                    continue
                plot_wst(data_fn, directory, fn)