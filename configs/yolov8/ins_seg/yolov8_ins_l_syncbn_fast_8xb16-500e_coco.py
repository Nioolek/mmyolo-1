_base_ = './yolov8_ins_s_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

# ===============================Unmodified in most cases====================
model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        type='YOLOv8InsHead',
        head_module=dict(
            type='YOLOv8InsHeadModule', masks_channels=32,
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels],
            protos_channels=256)))

