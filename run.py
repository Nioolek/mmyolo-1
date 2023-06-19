import os
import mmcv
import mmengine

from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmyolo.utils import register_all_modules


register_all_modules()


dataset = YOLOv5CocoDataset(
    data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(1600, 1600),
                use_cached=True,
                max_cached_images=40,
                pad_val=114.0),
            dict(
                type='mmdet.RandomResize',
                scale=(3200, 3200),
                ratio_range=(0.1, 2.0),
                resize_type='mmdet.Resize',
                keep_ratio=True),
            dict(type='mmdet.RandomCrop', crop_size=(1600, 1600)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.Pad',
                size=(1600, 1600),
                pad_val=dict(img=(114, 114, 114))),
            dict(type='CopyPasteIJCAI', cache_num=20),
            dict(type='YOLOv5MixUp', use_cached=True, max_cached_images=20),
            dict(type='mmdet.PackDetInputs')
        ],
        metainfo=dict(
            classes=('battery', 'pressure', 'umbrella', 'OCbottle',
                     'glassbottle', 'lighter', 'electronicequipment', 'knife',
                     'metalbottle'),
            palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                     (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                     (0, 0, 192)]))

for ind, i in enumerate(dataset):
#     if i['inputs'].shape[1] != 1600 or i['inputs'].shape[2] != 1600:
    print(ind, i['inputs'].shape)