import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_INPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_all.pickle")
_DEFAULT_OUTPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_violin_stats.pickle")

# y range for plotting. Restricting KDE evaluation to this range avoids computing meaningless tails
YLIM = (0.0, 50.0)
KDE_GRID_N = 256        # number of KDE evaluation points (smoothness of the plot)
SUBSAMPLE_MAX = 7790723  # max subsample size for KDE; the shape barely changes beyond this
RNG_SEED = 0


def main():
    parser = argparse.ArgumentParser(description='Compute KDE statistics for violin plots')
    parser.add_argument('--input', default=_DEFAULT_INPUT, help='Path to kp3d_err_all.pickle')
    parser.add_argument('--output', default=_DEFAULT_OUTPUT, help='Path to the output pickle')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, 'rb') as f:
        obj = pickle.load(f)

    kp = np.asarray(obj['kp3d_err'], dtype=float)  # (20, totalFrame)

    if kp.ndim != 2 or kp.shape[0] != 20:
        raise ValueError(f'Unexpected shape: {kp.shape} (expected (20, totalFrame))')

    rng = np.random.default_rng(RNG_SEED)
    grid = np.linspace(YLIM[0], YLIM[1], KDE_GRID_N)

    results = []
    for i in range(kp.shape[0]):
        x = kp[i, ~np.isnan(kp[i])]

        if x.size == 0:
            results.append(None)
            continue

        # Statistics are computed from all data (not the subsample)
        stats = {
            'n':      int(x.size),
            'mean':   float(np.mean(x)),
            'median': float(np.median(x)),
            'q1':     float(np.percentile(x, 25)),
            'q3':     float(np.percentile(x, 75)),
            'min':    float(np.min(x)),
            'max':    float(np.max(x)),
            'std':    float(np.std(x)),
        }

        # Subsample only for KDE
        if x.size > SUBSAMPLE_MAX:
            idx = rng.choice(x.size, SUBSAMPLE_MAX, replace=False)
            x_kde = x[idx]
        else:
            x_kde = x

        kde = gaussian_kde(x_kde)
        density = kde(grid)

        results.append({
            'kp_index':    i,
            'stats':       stats,
            'kde_grid':    grid,
            'kde_density': density,
        })
        print(f'kp{i:02d}: n={stats["n"]:>8d}  '
              f'mean={stats["mean"]:.3f}  median={stats["median"]:.3f}')

    with open(args.output, 'wb') as f:
        pickle.dump({
            'results': results,
            'meta': {'ylim': YLIM, 'kde_grid_n': KDE_GRID_N,
                     'subsample_max': SUBSAMPLE_MAX, 'seed': RNG_SEED},
        }, f)

    print(f'\nSaved: {args.output}')


if __name__ == "__main__":
    main()
