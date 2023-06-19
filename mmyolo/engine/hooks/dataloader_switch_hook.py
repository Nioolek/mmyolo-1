# Copyright (c) OpenMMLab. All rights reserved.
import itertools

from mmcv.transforms import Compose
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmyolo.registry import HOOKS, MODELS


@HOOKS.register_module()
class DataloaderSwitchHook(Hook):
    def __init__(self, switch_epoch, switch_dataloader, switch_data_preprocessor):
        self.switch_epoch = switch_epoch
        self.switch_dataloader = switch_dataloader
        self.switch_data_preprocessor = switch_data_preprocessor
        self._restart_dataloader = False
        self.is_switch = False

    def before_train_epoch(self, runner):
        """switch pipeline."""
        # epoch是从0开始的
        epoch = runner.epoch
        # train_loader = runner.train_dataloader
        if epoch >= self.switch_epoch and (self.is_switch is False):
            runner.logger.info('Switch dataloader and data_preprocessor now!')

            # 更新dataloader
            if isinstance(self.switch_dataloader, dict):
                # Determine whether or not different ranks use different seed.
                diff_rank_seed = runner._randomness_cfg.get(
                    'diff_rank_seed', False)
                new_dataloader = runner.build_dataloader(
                    self.switch_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
                runner.train_loop.dataloader = new_dataloader
                if hasattr(new_dataloader, 'persistent_workers'
                           ) and new_dataloader.persistent_workers is True:
                    new_dataloader._DataLoader__initialized = False
                    new_dataloader._iterator = None
                    self._restart_dataloader = True
                print('new dataset len', len(new_dataloader.dataset))
            else:
                runner.train_loop.dataloader = self.switch_dataloader

            rank, world_size = get_dist_info()

            # 更新dataprecoss
            model = runner.model
            if world_size > 1:
                model = model.module
            model.data_preprocessor = MODELS.build(self.switch_data_preprocessor)


            # ema更新参数
            print('将teacher模型参数copy给student')

            if world_size > 1:
                ema_model = runner.model.module.teacher
                src_model = runner.model.module.student
            else:
                ema_model = runner.model.teacher
                src_model = runner.model.student

            avg_param = (
                itertools.chain(ema_model.module.parameters(),
                                ema_model.module.buffers())
                if ema_model.update_buffers else
                ema_model.module.parameters())
            src_param = (
                itertools.chain(src_model.parameters(),
                                src_model.buffers())
                if ema_model.update_buffers else src_model.parameters())
            for p_avg, p_src in zip(avg_param, src_param):
                p_src.data.copy_(p_avg.data)
                # tmp = p_avg.data.clone()
                # p_avg.data.copy_(p_src.data)
                # p_src.data.copy_(tmp)
            print('copy结束')

            # 将开始半监督学习的标志打开
            model.unsup_training = True
            self.is_switch = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                runner.train_dataloader._DataLoader__initialized = True

