_base_ = './rtmdet_x_comp_1600_p5_dfl.py'


# =======================Unmodified in most cases==================
model = dict(
    bbox_head=dict(
        loss_dfl=dict(
             _delete_=True,
             type='mmdet.DistributionFocalLoss',
             reduction='mean',
             loss_weight=1.0 / 4)))
