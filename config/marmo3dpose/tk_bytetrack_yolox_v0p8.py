_base_ = ['../../scripts/03_run_marmo3dpose/marmo3Dpose/model/track/marmo20/tk_bytetrack_yolox_v0p8.py']

# Override checkpoint path when running from the ez2use root
model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='3D_digital_twin_marmoset_data/marmoset_keypoint_data/trained_model/detection.pth'
        )))
