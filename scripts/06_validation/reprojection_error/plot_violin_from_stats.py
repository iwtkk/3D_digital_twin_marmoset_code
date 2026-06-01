import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_INPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_violin_stats.pickle")
_DEFAULT_OUTPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_violin_from_stats.png")

KP_NAMES = [
    'nose', 'L eye', 'R eye', 'L ear', 'R ear',
    'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
    'L hip', 'R hip', 'L knee', 'R knee', 'L ankle', 'R ankle',
    'backbone', 'tail1', 'tail2',
]


def main():
    parser = argparse.ArgumentParser(description='事前計算済み統計量からバイオリンプロットを表示する')
    parser.add_argument('--input', default=_DEFAULT_INPUT, help='kp3d_err_violin_stats.pickleのパス')
    parser.add_argument('--output', default=_DEFAULT_OUTPUT, help='出力PNGのパス')
    parser.add_argument('--show', action='store_true', help='保存後にウィンドウ表示する')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        payload = pickle.load(f)
    results = payload['results']
    ylim = payload['meta']['ylim']

    WIDTH = 0.4  # 各 violin の最大半幅

    fig, ax = plt.subplots(figsize=(14, 5))

    # 全 violin をローカルに正規化（=各キーポイント内で最大密度を WIDTH に合わせる）
    for i, r in enumerate(results):
        if r is None:
            continue
        pos = i + 1
        grid = r['kde_grid']
        dens = r['kde_density']

        if dens.max() > 0:
            w = dens / dens.max() * WIDTH
        else:
            w = dens

        ax.fill_betweenx(grid, pos - w, pos + w,
                         facecolor='C0', edgecolor='C0', alpha=0.45, lw=0.8)

        s = r['stats']
        ax.hlines(s['median'], pos - WIDTH * 0.7, pos + WIDTH * 0.7,
                  colors='#BF0000', lw=1.6,
                  label='median' if i == 0 else None)

    ax.set_xticks(np.arange(1, len(results) + 1))
    ax.set_xticklabels(KP_NAMES[:len(results)], fontsize=25, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=25)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'saved to {output_path}')
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
