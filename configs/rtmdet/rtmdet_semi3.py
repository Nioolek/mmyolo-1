_base_ = './rtmdet_x_comp_1600_p5_dfl.py'

# 先是网络部分
detector = _base_.model

# 半监督用mmdet的，全监督用mmyolo的datapreprocessor
data_preprocessor = dict(type='SemiDataPreprocessor')
detector.data_preprocessor = data_preprocessor

model = dict(
    _delete_=True,
    type='EfficientTeacher',
    # 这里是efficientteacher用的data_preprocessor
    data_preprocessor=data_preprocessor,
    detector=detector,
    semi_train_cfg=dict(
        # freeze_teacher=True,
        # sup_weight=1.0,
        # unsup_weight=4.0,
        # pseudo_label_initial_score_thr=0.5,
        # rpn_pseudo_thr=0.9,
        # cls_pseudo_thr=0.9,
        # reg_pseudo_thr=0.02,
        # jitter_times=10,
        # jitter_scale=0.06,
        # min_pseudo_bbox_wh=(1e-2, 1e-2)
    ),
    semi_test_cfg=dict(predict_on='teacher'))

# 再是数据部分
dataset_type = _base_.dataset_type
data_root = _base_.data_root
img_scale = (1600-64-64, 1600-64-64)  # width, height
batch_size = _base_.train_batch_size_per_gpu
semi_batch_size = 2
num_workers = 10

branch_field = ['sup', 'unsup_teacher', 'unsup_student']

sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=_base_.mosaic_max_cached_images,
        pad_val=114.0),
    # dict(
    #     type='RandomResizeYOLO',
    #     # img_scale is (width, height)
    #     scale=(img_scale[0] * 2, img_scale[1] * 2),
    #     ratio_range=_base_.random_resize_ratio_range,
    #     resize_type='mmdet.Resize',
    #     keep_ratio=True),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=_base_.random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    # dict(type='RandomCropYOLO', crop_size=img_scale, allow_negative_crop=True),
    dict(type='RandomCropYOLO', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='vertical'),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    # mixup伪标签怎么融合的要看下
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=_base_.mixup_max_cached_images),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        sup=dict(type='mmdet.PackDetInputs')),
    # 由于上面有了multibranch，这里就不需要packdetinputs了
    #dict(type='mmdet.PackDetInputs')
]

weak_pipeline = [
    dict(type='SemiOrgimg2img'),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

strong_pipeline = [
    dict(type='RandomFlipYOLO'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.CutOut', n_holes=(0, 3), cutout_shape=(32, 32)),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                                'flip_state', 'matrix', 'scaleing_affine'))
]

unsup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadEmptyAnnotations'),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=_base_.mosaic_max_cached_images,
        pad_val=114.0),     # (3200, 3200)
    dict(type='RandomCropYOLO', crop_size=img_scale, allow_negative_crop=True,),   # (1600, 1600)
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5RandomAffine',     # ori_img (1600, 1600) img (1600, 1600)
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - 0.2, 1 + 0.2),
        # img_scale is (width, height)
        border=(0, 0),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

# 前面这些epoch使用原本的dataloader
train_dataloader = _base_.train_dataloader

labeled_dataset = dict(
    type=dataset_type,
    metainfo=_base_.metainfo,
    data_root=data_root,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=_base_.backend_args)

unlabeled_dataset = dict(
    type=dataset_type,
    metainfo=_base_.metainfo,
    data_root=data_root,
    ann_file='Semi-supervision/instances_new_test.json',
    data_prefix=dict(img='Semi-supervision/images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=_base_.backend_args)

# 半监督学习的dataloader
train_dataloader_semi = dict(
    batch_size=batch_size+semi_batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type='SemiMultiSampler', shuffle=True),
    batch_sampler=dict(
        type='SemiBatchSampler',
        drop_last=True,
        unsup_num=1000,
        start_num=1799,
        sup_batch_size=batch_size,
        semi_batch_size=semi_batch_size
    ),
    # sampler=dict(
    #     type='SemiSampler',
    #     batch_size=batch_size,
    #     source_ratio=[1, 1]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))

# test_dataloader和val_dataloader直接用base的就行

# 这边待确认，v5data_preprocessor 适不适用
semi_data_preprocessor = data_preprocessor

custom_hooks = [
    dict(
        type='EfficientTeacherHook',
        priority=49),
    dict(
        type='DataloaderSwitchHook',
        # switch_epoch=120,
        # TODO: 先训练一个模型，用于测试
        switch_epoch=110,
        switch_dataloader=train_dataloader_semi,
        # 貌似可以删除
        switch_data_preprocessor=semi_data_preprocessor)
]

val_cfg = dict(type='mmdet.TeacherStudentValLoop')
