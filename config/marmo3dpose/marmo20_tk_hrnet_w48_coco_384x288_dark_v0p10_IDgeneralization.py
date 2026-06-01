_base_ = ['../../scripts/03_run_marmo3dpose/marmo3Dpose/model/pose/marmo20/marmo20_tk_hrnet_w48_coco_384x288_dark_v0p10_IDgeneralization.py']

# ez2useルートから実行する際のload_fromパス上書き
load_from = '3D_digital_twin_marmoset_data/marmoset_keypoint_data/trained_model/pose_marmo20_best_epoch.pth'
