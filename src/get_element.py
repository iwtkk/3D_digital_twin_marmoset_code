import numpy as np
import pandas as pd
from utils import file_utls, frame_to_imgs
import json

def element_search_from_data(arr):
    T, K = arr.shape

    # その列に一つでも有効値(非NaN)があるか
    valid_any = np.any(~np.isnan(arr), axis=0)

    # デフォは -1（=有効値がない列の印）
    idx_max = np.full(K, -1, dtype=int)
    idx_min = np.full(K, -1, dtype=int)

    # 有効な列だけ nanargmax / nanargmin
    if np.any(valid_any):
        idx_max[valid_any] = np.nanargmax(arr[:, valid_any], axis=0)
        idx_min[valid_any] = np.nanargmin(arr[:, valid_any], axis=0)

    # 対応する値（-1 の列は NaN にしておく）
    max_val = np.where(idx_max >= 0, arr[idx_max, np.arange(K)], np.nan)
    min_val = np.where(idx_min >= 0, arr[idx_min, np.arange(K)], np.nan)
    
    # 結果を表に
    result = pd.DataFrame({
        "joint_id": np.arange(K),
        "max_frame_idx": idx_max,
        "max_value": max_val,
        "min_frame_idx": idx_min,
        "min_value": min_val,
    })

    # print(result.head(20).round(5))

    return result

def data_to_element(file, key='kp3d_err'):
    data = file_utls.open_pickle(file)
    
    print(data.keys())

    kp3d_err = data[key]

    kp3d_err = np.asarray(kp3d_err)

    
    # 先頭の次元(=1)とフレーム次元をまとめて (N, 20) に整形
    if kp3d_err.ndim == 3 and kp3d_err.shape[-1] == 20:
        arr = kp3d_err.reshape(-1, kp3d_err.shape[-1])  # (28755, 20)
    elif kp3d_err.ndim == 2 and kp3d_err.shape[-1] == 20:
        arr = kp3d_err
    else:
        raise ValueError(f"予期しない形状: {kp3d_err.shape}")

    # NaNを無視して列(=各要素/関節)ごとに統計量を計算
    mean   = np.nanmean(arr, axis=0)
    median = np.nanmedian(arr, axis=0)
    max_   = np.nanmax(arr, axis=0)
    min_   = np.nanmin(arr, axis=0)

    return arr, mean, median, max_, min_
    
    
def main():
    file = 'results/3d_v0p8_dark_fix_20/dailylife_cj611_20230226_110000/kp3d_fxdJointLen.pickle'
    arr, mean, median, max_, min_ = data_to_element(file)
    result = element_search_from_data(arr)
    # 表にまとめる（0〜19を関節IDとしておく）
    df = pd.DataFrame({
        # "joint_id": np.arange(arr.shape[1]),
        "mean": mean,
        "median": median,
        "max": max_,
        "min": min_,
    })

    print(df.round(4))
    
    ann_path = './utils/tkmarmoV0p42.json'
    # print(result.keys())
    result_keys = result.keys()
    min_frame_idxs =result[result_keys[3]]
    min_frame = min_frame_idxs[15]
    print(arr.shape)
    print(arr[99])
    print(arr[99][9])
    # frame_to_imgs.frame_to_imgs(min_frame, output_dir='./es')
    
    data = file_utls.open_json(ann_path)
    print(data.keys())
    keypoints_names = data['categories'][0]['keypoints']
    print(keypoints_names[9])
    
    import os
    jsonfile = 'results/2d_v0p8_Dark_fix_20/dailylife_cj611_20230226_110000'
    jsonlist = os.listdir(jsonfile)
    
    print(jsonlist)
    pose = file_utls.open_pickle(file)
    pose = pose['kp3d']
    print(np.asarray(pose)[0].shape)
       
if '__main__' == __name__:
    main()
    