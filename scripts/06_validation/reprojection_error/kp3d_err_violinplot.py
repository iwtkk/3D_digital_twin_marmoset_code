import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_INPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_all.pickle")
_DEFAULT_OUTPUT = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error" / "kp3d_err_violinplot.png")


def main():
    parser = argparse.ArgumentParser(description='Show a violin plot of kp3d_err')
    parser.add_argument('--input', default=_DEFAULT_INPUT, help='Path to kp3d_err_all.pickle')
    parser.add_argument('--output', default=_DEFAULT_OUTPUT, help='Path to the output PNG')
    parser.add_argument('--show', action='store_true', help='Show a window after saving')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        obj = pickle.load(f)

    kp = np.asarray(obj['kp3d_err'], dtype=float)  # expected: (20, totalFrame)

    if kp.ndim != 2 or kp.shape[0] != 20:
        raise ValueError(f'Unexpected shape: {kp.shape} (expected (20, totalFrame))')

    # Drop NaNs and build lists for the violin plot (one per keypoint, 20 total)
    data_list = [kp[i, ~np.isnan(kp[i])] for i in range(20)]

    plt.figure(figsize=(14, 5))

    plt.violinplot(
        data_list,
        positions=np.arange(1, 21),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

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
