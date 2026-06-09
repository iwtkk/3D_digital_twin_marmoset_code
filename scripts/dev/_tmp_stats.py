import json, numpy as np, sys
from pathlib import Path

OUT_DIR = Path('outputs/analysis_kp3d_outliers')
files = sorted(OUT_DIR.glob('*_outlier.json'))

devs = []
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    devs.extend(ev['deviation_mm'] for ev in d['outlier_events'])

devs = np.array(devs)
print(f'N total events: {len(devs):,}', flush=True)
for p in [95,96,97,97.5,98,98.5,99,99.1,99.5]:
    print(f'p{p}: {np.percentile(devs,p):.2f} mm', flush=True)
print(flush=True)
bins = [0,10,20,50,100,200,500,1000,5000,15000]
counts, _ = np.histogram(devs, bins=bins)
for lo, hi, cnt in zip(bins, bins[1:], counts):
    print(f'  [{lo:>6}, {hi:>6}) mm : {cnt:>10,}  ({cnt/len(devs)*100:.2f}%)', flush=True)
sys.stdout.flush()
