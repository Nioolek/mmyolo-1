import torch
from mmengine import Config

from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/rtmdet/rtmdet-ins_l_syncbn_fast_8xb32-300e_coco.py')
model = MODELS.build(cfg.model)

state_dict = torch.load('rtmdet_ins_l_mmyolo.pth')

print([i for i in model.state_dict().keys() if i.startswith('bbox_head')])
# print([i for i in state_dict['state_dict'].keys() if i.startswith('bbox_head')])

# print(state_dict['state_dict'])
