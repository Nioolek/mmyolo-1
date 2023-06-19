_base_ = './rtmdet_x_comp_1280_p5_dfl.py'

random_resize_ratio_range = (0.5, 2.0)

train_pipeline = _base_.train_pipeline

train_pipeline[3] = dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(_base_.img_scale[0] * 2, _base_.img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True)

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline))
