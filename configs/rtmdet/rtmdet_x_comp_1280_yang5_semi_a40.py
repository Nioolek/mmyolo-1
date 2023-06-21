_base_ = './rtmdet_x_comp_1280_yang5.py'

# 相比较yang3_semi_a40添加的内容：
# 1、数据增强增加random flip vertical
# 2、数据增强增加randomrotate
# 3、copypaste1里增加randomrotate
# 4、修复copypaste1失败的情况
# 5、efficient teacher里，nms改成softnms
# 6、switch从180改为200

# 待添加事项：
# 1、半监督预先加载一些标注数据
# 2、

# 先是网络部分
detector = _base_.model

# 半监督用mmdet的，全监督用mmyolo的datapreprocessor
data_preprocessor = dict(type='SemiDataPreprocessor')
detector.data_preprocessor = data_preprocessor
detector.bbox_head.use_semi_uncertain = False
max_epochs = 300

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
img_scale = (1408, 1408)  # width, height
batch_size = 2
semi_batch_size = batch_size
num_workers = 10
base_lr = 0.004 / 2

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
    dict(
        type='CopyPasteIJCAI1',
        n=(-0.1, 0.6)
    ),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=_base_.random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='RandomCropYOLO', crop_size=img_scale),
    dict(
        type='mmdet.Albu',
        transforms=_base_.albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='vertical'),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomRotateYOLO'),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=_base_.mixup_max_cached_images),
    dict(
        type='mmdet.MultiBranch',
        branch_field=branch_field,
        sup=dict(type='mmdet.PackDetInputs'))
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
    dict(type='RandomCropYOLO', crop_size=img_scale, allow_negative_crop=True),   # (1600, 1600)
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

train_dataloader = dict(
    batch_size=batch_size,
    # num_workers=0,
    # persistent_workers=False,
    dataset=dict(
        metainfo=_base_.metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline))

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

semi_data_preprocessor = data_preprocessor

# optimizer
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=base_lr))

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=_base_.lr_start_factor,
    #     by_epoch=False,
    #     begin=0,
    #     end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,
        begin=0,
        end=max_epochs,
        T_max=0,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EfficientTeacherHook',
        priority=49),
    dict(
        type='DataloaderSwitchHook',
        switch_epoch=200,
        switch_dataloader=train_dataloader_semi,
        # 貌似可以删除
        switch_data_preprocessor=semi_data_preprocessor)
]

# val and test switch
# test_image_info = 'annotations/instances_test2017.json'
# test_image = 'test2017/'

test_image_info = 'annotations/test_phase2.json'
test_image = 'test_phase2/images/'

_test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        metainfo=_base_.metainfo,
        ann_file=test_image_info,
        data_prefix=dict(img=test_image),
        test_mode=True,
        batch_shapes_cfg=_base_.batch_shapes_cfg,
        pipeline=_base_.test_pipeline))

_test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 300, 1000),
    ann_file=_base_.data_root + test_image_info,
    metric='bbox',
    format_only=True,  # 只将模型输出转换为coco的 JSON 格式并保存
    outfile_prefix='test-date0620_softnms065_175e',  # 要保存的 JSON 文件的前缀
)

# # Reduce evaluation time
# val_evaluator = dict(
#     # type='mmdet.CocoMetric',
#     # proposal_nums=(100, 300, 1000),
#     # # ann_file=data_root + val_ann_file,
#     # metric='bbox',
#     format_only=True,  # 只将模型输出转换为coco的 JSON 格式并保存
#     outfile_prefix='val-date0620_175e_nms095'
# )
# test_evaluator = val_evaluator

test_dataloader = _test_dataloader
test_evaluator = _test_evaluator
