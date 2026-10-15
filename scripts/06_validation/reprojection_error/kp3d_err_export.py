"""
Convert kp3d_err of all sessions to mat/pickle format

Merges kp3d_err_tomat.py and kp3d_err_topickle.py into one script.
Use --format to choose the output format.
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_BASE = str(_PROJECT_ROOT / "3D_digital_twin_marmoset_data" / "marmoset_keypoint_data" / "3D_temporal_keypoint_data")
_DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "outputs" / "validation" / "reprojection_error")


def main():
    parser = argparse.ArgumentParser(description='Convert kp3d_err of all sessions to mat/pickle')
    parser.add_argument('--base', default=_DEFAULT_BASE, help='Root directory of the 3D keypoint data')
    parser.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--sessions', nargs='*', default=None, help='Session names to process. All sessions if omitted')
    parser.add_argument('--format', choices=['mat', 'pickle', 'both'], default='both',
                        help='Output format (default: both)')
    args = parser.parse_args()

    result_path = args.base
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = args.sessions or sorted(glob.glob('*', root_dir=result_path))

    kp3d_err_list = []

    for session in sessions:
        picklefile = os.path.join(result_path, session, 'kp3d_fxdJointLen.pickle')

        if not os.path.isfile(picklefile):
            print(f'[WARN] skip (not found): {picklefile}')
            continue

        with open(picklefile, 'rb') as f:
            data = pickle.load(f)

        # Expected shape: (nAnimal=1, nFrame, nKeypoints=20)
        kp3d_err = np.array(data['kp3d_err'], dtype=float)
        print(session, 'kp3d_err.shape =', kp3d_err.shape)

        if kp3d_err.ndim != 3 or kp3d_err.shape[0] != 1:
            print(f'[WARN] unexpected shape {kp3d_err.shape} in {session}, skip')
            continue

        kp3d_err_list.append(kp3d_err[0])  # (nFrame, 20)

    if len(kp3d_err_list) == 0:
        raise RuntimeError('No kp3d_err data loaded. Check result_path or file names.')

    # Concatenate along frames: (totalFrame, 20) -> transpose to (20, totalFrame)
    kp3d_err_all = np.concatenate(kp3d_err_list, axis=0)
    kp3d_err_all_T = kp3d_err_all.T
    print('kp3d_err_all_T (20, totalFrame) =', kp3d_err_all_T.shape)
    print('has NaN:', np.isnan(kp3d_err_all_T).any())
    print('num NaN:', np.isnan(kp3d_err_all_T).sum())
    print('max:', np.nanmax(kp3d_err_all_T))
    print('min:', np.nanmin(kp3d_err_all_T))
    print('mean:', np.nanmean(kp3d_err_all_T))

    if args.format in ('mat', 'both'):
        from scipy.io import savemat
        out_path = output_dir / 'kp3d_err_all.mat'
        savemat(str(out_path), {'kp3d_err': kp3d_err_all_T})
        print(f'saved to {out_path}')

    if args.format in ('pickle', 'both'):
        out_path = output_dir / 'kp3d_err_all.pickle'
        with open(out_path, 'wb') as f:
            pickle.dump({'kp3d_err': kp3d_err_all_T}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saved to {out_path}')


if __name__ == "__main__":
    main()
