import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_violin_stats import robust_bw_factor  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

KDE_SUBSAMPLE = 1_000_000   # KDE shape only; mean/median use all values

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_INPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_all.pickle")
_DEFAULT_OUTPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_violinplot.png")


def main():
    parser = argparse.ArgumentParser(description='kp3d_errのバイオリンプロットを表示する')
    parser.add_argument('--input', default=_DEFAULT_INPUT, help='kp3d_err_all.pickleのパス')
    parser.add_argument('--output', default=_DEFAULT_OUTPUT, help='出力PNGのパス')
    parser.add_argument('--show', action='store_true', help='保存後にウィンドウ表示する')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        obj = pickle.load(f)

    kp = np.asarray(obj['kp3d_err'], dtype=float)  # 期待: (20, totalFrame)

    if kp.ndim != 2 or kp.shape[0] != 20:
        raise ValueError(f'Unexpected shape: {kp.shape} (expected (20, totalFrame))')

    # NaN を落として violin 用のリストにする（20個ぶん）
    data_list = [kp[i, ~np.isnan(kp[i])] for i in range(20)]

    plt.figure(figsize=(14, 5))

    # plt.violinplot evaluates the KDE on 100 points between min and max of each keypoint; with failed
    # triangulations (max ~6e5 px) almost none of them fall in the plotted 0-50 px. Evaluate on the plotted
    # range instead, with the robust bandwidth of compute_violin_stats (KDE on a subsample, stats on all data).
    rng = np.random.default_rng(0)
    coords = np.linspace(0.0, 50.0, 256)
    vpstats = []
    for x in data_list:
        xs = x if x.size <= KDE_SUBSAMPLE else rng.choice(x, KDE_SUBSAMPLE, replace=False)
        kde = gaussian_kde(xs, bw_method=robust_bw_factor(xs))
        vpstats.append({'coords': coords, 'vals': kde(coords), 'mean': float(np.mean(x)),
                        'median': float(np.median(x)), 'min': float(np.min(x)), 'max': float(np.max(x)),
                        'quantiles': []})
    plt.gca().violin(vpstats, positions=np.arange(1, 21), showmeans=False, showmedians=True, showextrema=False)

    plt.xticks(np.arange(1, 21), [str(i) for i in range(20)])
    plt.xlabel('Keypoint index (0-19)')
    plt.ylabel('kp3d_err')
    plt.ylim(0, 50)
    plt.title('Violin plot of kp3d_err per keypoint')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'saved to {output_path}')
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
