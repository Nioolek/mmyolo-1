_base_ = './rtmdet_l_syncbn_fast_8xb32-300e_coco.py'

# ========================modified parameters======================
deepen_factor = 1.33
widen_factor = 1.25

num_classes = 9

metainfo = dict(
    classes=('battery', 'pressure', 'umbrella', 'OCbottle', 'glassbottle',
             'lighter', 'electronicequipment', 'knife', 'metalbottle'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
             (0, 0, 192)])

train_batch_size_per_gpu = 2
val_batch_size_per_gpu = 1

base_lr = 0.004 / 8
max_epochs = 300  # Maximum training epochs

img_scale = (1280, 1280)  # width, height

# Config of batch shapes. Only on val.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 5

# validation intervals in stage 2
val_interval_stage2 = 1

# The maximum checkpoints to keep.
max_keep_ckpts = 5

random_resize_ratio_range = (0.5, 2.0)

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='soft_nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes, widen_factor=widen_factor)),
    train_cfg=dict(assigner=dict(dict(num_classes=num_classes))),
    test_cfg=model_test_cfg)
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
train_pipeline = [
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
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='RandomCropYOLO', crop_size=img_scale),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
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
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=_base_.mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    # num_workers=0,
    # persistent_workers=False,
    dataset=dict(
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        metainfo=metainfo,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))

test_dataloader = val_dataloader
# test_evaluator = _base_.val_evaluator

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

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts,  # only keep latest 5 checkpoints
        save_best='coco/bbox_mAP_50',
        rule='greater',
    ))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals,
    dynamic_intervals=[(max_epochs - _base_.num_epochs_stage2,
                        val_interval_stage2)])

img_scales = [(int(img_scale[0] * i), int(img_scale[1] * i))
              for i in (1, 0.5, 1.5)]

_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in img_scales
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]

# runtime settings
load_from = 'weights/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345-b85cd476.pth'  # noqa
resume = False

# val and test switch
test_image_info = 'annotations/instances_test2017.json'
test_image = 'test2017/'

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
        metainfo=metainfo,
        ann_file=test_image_info,
        data_prefix=dict(img=test_image),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))

_test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 300, 1000),
    ann_file=_base_.data_root + test_image_info,
    metric='bbox',
    format_only=True,  # 只将模型输出转换为coco的 JSON 格式并保存
    outfile_prefix='test-xxxx',  # 要保存的 JSON 文件的前缀
)

# test_dataloader = _test_dataloader
# test_evaluator = _test_evaluator
