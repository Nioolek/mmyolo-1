_base_ = './rtmdet_x_comp_1280_yang3.py'


# =======================Unmodified in most cases==================
model = dict(
    bbox_head=dict(
        type='RTMDetHeadDFL',
        head_module=dict(
            type='RTMDetSepBNHeadModuleDFL',
            # 配置1，优先训练
            reg_max=24,
            dfl_scale=1.0,
            # 配置2,
            # reg_max=16,
            # dfl_scale=2.0
        ),
        loss_dfl=None
    ))
