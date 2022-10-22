_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

deepen_factor = 0.67
widen_factor = 0.75

model = dict(
    backbone=dict(
        type='CSPResNet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
