_base_ = './fold_base.py'

train_ann_file = 'annotations/instances_train_fold_7.json'
val_ann_file = 'annotations/instances_val_fold_7.json'


semi_ann_file = 'Semi-supervision/instances_new_test.json'
unsup_num = 1000

train_dataloader = _base_.train_dataloader
train_dataloader.dataset.ann_file = train_ann_file

val_dataloader = _base_.val_dataloader
val_dataloader.dataset.ann_file = val_ann_file

labeled_dataset = _base_.labeled_dataset
labeled_dataset.ann_file = train_ann_file

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.ann_file = semi_ann_file
# 半监督学习的dataloader
train_dataloader_semi = dict(
    batch_size=_base_.batch_size+_base_.semi_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=False,
    sampler=dict(type='SemiMultiSampler', shuffle=True),
    batch_sampler=dict(
        type='SemiBatchSampler',
        drop_last=True,
        unsup_num=unsup_num,
        start_num=1800,
        sup_batch_size=_base_.batch_size,
        semi_batch_size=_base_.semi_batch_size
    ),
    # sampler=dict(
    #     type='SemiSampler',
    #     batch_size=batch_size,
    #     source_ratio=[1, 1]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))
# train_dataloader_semi = dict(
#     batch_sampler=dict(unsup_num=unsup_num),
#     dataset=dict(
#         datasets=[labeled_dataset, unlabeled_dataset]))


semi_data_preprocessor = _base_.semi_data_preprocessor
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

val_evaluator = dict(ann_file=_base_.data_root + val_ann_file)
test_evaluator = val_evaluator
test_dataloader = val_dataloader



# # # 阶段2
# test_image_info = 'annotations/test_phase2.json'
# test_image = 'test_phase2/images/'
#
# _test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     pin_memory=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=_base_.dataset_type,
#         data_root=_base_.data_root,
#         metainfo=_base_.metainfo,
#         ann_file=test_image_info,
#         data_prefix=dict(img=test_image),
#         test_mode=True,
#         batch_shapes_cfg=_base_.batch_shapes_cfg,
#         pipeline=_base_.test_pipeline))
#
# _test_evaluator = dict(
#     type='mmdet.CocoMetric',
#     proposal_nums=(100, 300, 1000),
#     ann_file=_base_.data_root + test_image_info,
#     metric='bbox',
#     format_only=True,  # 只将模型输出转换为coco的 JSON 格式并保存
#     outfile_prefix='test2-0627_fold7',  # 要保存的 JSON 文件的前缀
# )
# test_dataloader = _test_dataloader
# test_evaluator = _test_evaluator
