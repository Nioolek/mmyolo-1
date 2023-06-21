import copy
import itertools
import logging
from typing import List, Optional, Tuple, Dict, Union

import mmcv
import torch
from mmcv.ops import batched_nms
from mmdet.models.utils import filter_scores_and_topk, unpack_gt_instances
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor, HorizontalBoxes
from mmengine import print_log
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models import SemiBaseDetector, DetDataPreprocessor, MultiBranchDataPreprocessor, DetTTAModel
from mmengine.structures import InstanceData
from torch import Tensor
import numpy as np

from mmyolo.registry import HOOKS
from mmyolo.registry import MODELS


@MODELS.register_module()
class EfficientTeacher(SemiBaseDetector):
    def __init__(self,
                 detector: ConfigType,
                 nms_conf_thres=0.1,
                 nms_iou_thres=0.65,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        del self.teacher
        # TODO: ema参数待填写
        # teacher在hook里初始化
        self.teacher = None
        self.init_teacher = False
        # 为False的时候是监督训练，设置为True就包含了半监督训练
        self.unsup_training = False

        # 为ssod生成伪标签，所用的变量
        self.ssod_test_cfg = {
            'multi_label': False,
            'nms_conf_thres': nms_conf_thres,
            'nms_iou_thres': nms_iou_thres,
            'type': 'soft_nms',
            'max_per_img': 300
        }
        self.featmap_sizes = None

        self.debug_count = 0

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # 如果是监督学习，那直接使用student逻辑就可以
        if self.unsup_training == False:
            # print('123123')
            log_vars = self.student.train_step(data, optim_wrapper)
        # 半监督学习就比较复杂
        else:
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                data_ = self.data_preprocessor(data, True)
                data_['org_data'] = data
                losses = self._run_forward(data_, mode='loss')  # type: ignore
            parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
            optim_wrapper.update_params(parsed_losses)

        return log_vars

    # def loss(self, multi_batch_inputs: Dict[str, Tensor],
    #          multi_batch_data_samples: Dict[str, SampleList], org_data=None) -> dict:
    #     if self.unsup_training is False:
    #         log_vars = self.student.loss(multi_batch_inputs, multi_batch_data_samples)
    #     else:
    #         # teacher为弱增强
    #         sup_inputs = multi_batch_inputs['sup'].cuda()
    #         unsup_teacher_inputs = multi_batch_inputs['unsup_teacher'].cuda()
    #         unsup_student_inputs = multi_batch_inputs['unsup_student'].cuda()
    #
    #         with torch.no_grad():
    #             cls_scores, bbox_preds = self.teacher.module._forward(unsup_teacher_inputs)
    #             pe_results_list = self.create_pseudo_label_online_with_gt(cls_scores, bbox_preds, unsup_teacher_inputs.shape,
    #                                                     multi_batch_data_samples['unsup_student'])
    #
    #         rank, world_size = get_dist_info()
    #         if True and (rank in [-1, 0]):
    #             if self.debug_count % 100 == 0:
    #                 print('debug')
    #                 import cv2
    #                 # print('org_data', org_data)
    #                 org_data_teacher = org_data['inputs']['unsup_teacher'][0]
    #                 org_data_student = org_data['inputs']['unsup_student'][0]
    #                 t_img = org_data_teacher.cpu().numpy().transpose(1, 2, 0)
    #                 t_draw_img = t_img.astype(np.uint8).copy()
    #                 s_img = org_data_student.cpu().numpy().transpose(1, 2, 0)
    #                 s_draw_img = s_img.astype(np.uint8).copy()
    #
    #                 bboxes = pe_results_list[0].bboxes.cpu().numpy()
    #                 labels = pe_results_list[0].labels.cpu().numpy()
    #                 scores = pe_results_list[0].scores.cpu().numpy()
    #                 for i in range(len(bboxes)):
    #                     xmin, ymin, xmax, ymax = [int(k) for k in list(bboxes[i])]
    #                     lb = int(labels[i])
    #                     score = float(scores[i])
    #                     name = '%s %.3f' % (lb, score)
    #
    #                     cv2.rectangle(t_draw_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
    #                     cv2.putText(t_draw_img, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0),2)
    #                 mmcv.imwrite(t_draw_img, 'test_teacher_%s.jpg' % self.debug_count)
    #
    #                 print(111)
    #
    #                 stu_ins = multi_batch_data_samples['unsup_student'][0].gt_instances
    #                 stu_bboxes = stu_ins.bboxes.cpu().numpy()
    #                 stu_labels = stu_ins.labels.cpu().numpy()
    #                 stu_scores = stu_ins.scores.cpu().numpy()
    #                 for i in range(len(stu_bboxes)):
    #                     xmin, ymin, xmax, ymax = [int(k) for k in list(stu_bboxes[i])]
    #                     lb = int(stu_labels[i])
    #                     score = float(stu_scores[i])
    #                     name = '%s %.3f' % (lb, score)
    #
    #                     cv2.rectangle(s_draw_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
    #                     cv2.putText(s_draw_img, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0),2)
    #                 mmcv.imwrite(s_draw_img, 'test_student_%s.jpg' % self.debug_count)
    #             self.debug_count += 1
    #
    #
    #         all_inputs = torch.cat([sup_inputs, unsup_student_inputs])
    #         sup_bs = len(sup_inputs)
    #
    #         # 分割出sup和unsup
    #         head_pred = self.student._forward(all_inputs)
    #         sup_pred = (tuple([k[:sup_bs] for k in head_pred[0]]), tuple([k[:sup_bs] for k in head_pred[1]]))
    #         unsup_pred = (tuple([k[sup_bs:] for k in head_pred[0]]), tuple([k[sup_bs:] for k in head_pred[1]]))
    #
    #         # sup loss
    #         outputs = unpack_gt_instances(multi_batch_data_samples['sup'])
    #         (batch_gt_instances, batch_gt_instances_ignore,
    #          batch_img_metas) = outputs
    #
    #         loss_inputs = sup_pred + ([i.cuda() for i in batch_gt_instances], batch_img_metas,
    #                               batch_gt_instances_ignore)
    #         losses = self.student.bbox_head.loss_by_feat(*loss_inputs)
    #
    #
    #         # semi sup loss
    #         semi_outputs = unpack_gt_instances(multi_batch_data_samples['unsup_student'])
    #         semi_batch_pesu_instances, semi_batch_pesu_instances_ignore, semi_batch_img_metas = semi_outputs
    #         semi_loss_inputs = unsup_pred + ([i.cuda() for i in semi_batch_pesu_instances], semi_batch_img_metas,
    #                                          semi_batch_pesu_instances_ignore)
    #         semi_losses = self.student.bbox_head.semi_loss_by_feat(*semi_loss_inputs)
    #         log_vars = {}
    #         log_vars.update(**losses)
    #         log_vars.update(**semi_losses)
    #
    #     # debug使用
    #
    #         # log_vars = dict(
    #         #     loss=log_vars['loss_cls']*0
    #         # )
    #         # print('debug loss', log_vars)
    #
    #     return log_vars


    # 这里的代码用于确认全部用监督学习会不会精度下降
    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList], org_data=None) -> dict:
        if self.unsup_training is False:
            # 多卡训练是直接调的这里
            # print('234235')
            log_vars = self.student.loss(multi_batch_inputs, multi_batch_data_samples)
        else:
            log_vars = dict()
            sup_log_vars = self.student.loss(multi_batch_inputs['sup'], multi_batch_data_samples['sup'])


            unsup_teacher_inputs = multi_batch_inputs['unsup_teacher']
            unsup_student_inputs = multi_batch_inputs['unsup_student']
            with torch.no_grad():
                cls_scores, bbox_preds = self.teacher.module._forward(unsup_teacher_inputs)
                pe_results_list = self.create_pseudo_label_online_with_gt(cls_scores, bbox_preds,
                                                                          unsup_teacher_inputs.shape,
                                                                          multi_batch_data_samples['unsup_student'])
                del cls_scores, bbox_preds, unsup_teacher_inputs

            semi_log_vars = self.student.semi_loss(unsup_student_inputs, multi_batch_data_samples['unsup_student'])
            log_vars.update(**sup_log_vars)
            log_vars.update(**semi_log_vars)

            # if False:
            #     rank, world_size = get_dist_info()
            #     if (rank in [-1, 0]) and (self.debug_count % 100 == 0):
            #         print('debug')
            #         import cv2
            #         # print('org_data', org_data)
            #         org_data_teacher = org_data['inputs']['unsup_teacher'][0]
            #         org_data_student = org_data['inputs']['unsup_student'][0]
            #         t_img = org_data_teacher.cpu().numpy().transpose(1, 2, 0)
            #         t_draw_img = t_img.astype(np.uint8).copy()
            #         s_img = org_data_student.cpu().numpy().transpose(1, 2, 0)
            #         s_draw_img = s_img.astype(np.uint8).copy()
            #
            #         bboxes = pe_results_list[0].bboxes.cpu().numpy()
            #         labels = pe_results_list[0].labels.cpu().numpy()
            #         scores = pe_results_list[0].scores.cpu().numpy()
            #         for i in range(len(bboxes)):
            #             xmin, ymin, xmax, ymax = [int(k) for k in list(bboxes[i])]
            #             lb = int(labels[i])
            #             score = float(scores[i])
            #             name = '%s %.3f' % (lb, score)
            #
            #             cv2.rectangle(t_draw_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
            #             cv2.putText(t_draw_img, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            #         mmcv.imwrite(t_draw_img, 'test_teacher_%s.jpg' % self.debug_count)
            #
            #         print(111)
            #
            #         stu_ins = multi_batch_data_samples['unsup_student'][0].gt_instances
            #         stu_bboxes = stu_ins.bboxes.cpu().numpy()
            #         stu_labels = stu_ins.labels.cpu().numpy()
            #         stu_scores = stu_ins.scores.cpu().numpy()
            #         for i in range(len(stu_bboxes)):
            #             xmin, ymin, xmax, ymax = [int(k) for k in list(stu_bboxes[i])]
            #             lb = int(stu_labels[i])
            #             score = float(stu_scores[i])
            #             name = '%s %.3f' % (lb, score)
            #
            #             cv2.rectangle(s_draw_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
            #             cv2.putText(s_draw_img, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            #         mmcv.imwrite(s_draw_img, 'test_student_%s.jpg' % self.debug_count)
            #     self.debug_count += 1

        return log_vars

    def predict_by_feat_ssod(self, cls_scores, bbox_preds):
        cfg = self.ssod_test_cfg
        multi_label = cfg['multi_label']
        multi_label &= self.student.bbox_head.num_classes > 1

        num_imgs = len(cls_scores[0])
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.student.bbox_head.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)
        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.teacher.module.bbox_head.num_base_priors,), stride) for
            featmap_size, stride in zip(featmap_sizes, self.teacher.module.bbox_head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.teacher.module.bbox_head.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.teacher.module.bbox_head.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        score_thr = self.ssod_test_cfg['nms_conf_thres']
        nms_pre = 30000
        results_list = []
        for (bboxes, scores) in zip(flatten_decoded_bboxes, flatten_cls_scores):
            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            if multi_label:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)
            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if results.bboxes.numel() > 0:
                bboxes = get_box_tensor(results.bboxes)
                det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                    results.labels,
                                                    {'iou_threshold': self.ssod_test_cfg['nms_iou_thres'],
                                                     'type': self.ssod_test_cfg['type']})
                results = results[keep_idxs]
                # some nms would reweight the score, such as softnms
                results.scores = det_bboxes[:, -1]
                results = results[:cfg['max_per_img']]
            results.scores = results.scores.detach()
            results.labels = results.labels.detach()
            results.bboxes = results.bboxes.detach()
            results_list.append(results)
        return results_list

    def filter_gt_bboxes(self, origin_bboxes: HorizontalBoxes,
                         wrapped_bboxes: HorizontalBoxes) -> torch.Tensor:
        """Filter gt bboxes.

        Args:
            origin_bboxes (HorizontalBoxes): Origin bboxes.
            wrapped_bboxes (HorizontalBoxes): Wrapped bboxes

        Returns:
            dict: The result dict.
        """
        origin_w = origin_bboxes.widths
        origin_h = origin_bboxes.heights
        wrapped_w = wrapped_bboxes.widths
        wrapped_h = wrapped_bboxes.heights
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > 20) & \
                       (wrapped_h > 20)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > 0.5
        aspect_ratio_valid_idx = aspect_ratio < 20
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    def create_pseudo_label_online_with_gt(self, cls_scores, bbox_preds, target_imgs_shape, student_datasample_list):
        n_img, _, height, width = target_imgs_shape
        results_list = self.predict_by_feat_ssod(cls_scores, bbox_preds)
        for i in range(n_img):
            student_datasample = student_datasample_list[i]
            flip_state = student_datasample.flip_state
            scaleing_affine = student_datasample.scaleing_affine
            matrix = student_datasample.matrix
            img_h, img_w, _ = student_datasample.img_shape

            # 将预测结果从teacher对应到student
            results = results_list[i]
            res_bboxes, res_labels, res_scores = results.bboxes, results.labels, results.scores
            hor_boxes = HorizontalBoxes(res_bboxes).cpu()
            hor_boxes_org = hor_boxes.clone()
            hor_boxes_org.rescale_([scaleing_affine, scaleing_affine])
            hor_boxes.project_(matrix)
            hor_boxes.clip_([img_h, img_w])
            valid_indexes = self.filter_gt_bboxes(hor_boxes_org, hor_boxes)

            # 获取保留的目标
            hor_boxes = hor_boxes[valid_indexes]
            res_labels = res_labels[valid_indexes]
            res_scores = res_scores[valid_indexes]

            if flip_state[0]:
                hor_boxes.flip_((img_h, img_w), direction='horizontal')
            if flip_state[1]:
                hor_boxes.flip_((img_h, img_w), direction='vertical')

            student_instances = InstanceData()
            student_instances.bboxes = hor_boxes.tensor.cuda()
            student_instances.labels = res_labels.cuda()
            student_instances.scores = res_scores.cuda()
            student_datasample.gt_instances = student_instances
        return results_list


    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        if not any([
            'student' in key or 'teacher' in key
            for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.module.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # teacher永远是eval模式
        self.teacher.eval()
        return self

    def forward(self,
                inputs: torch.Tensor,
                data_samples,
                org_data = None,
                mode: str = 'tensor'):

        if mode == 'loss':
            return self.loss(inputs, data_samples, org_data)
        elif mode == 'predict':
            inputs = inputs.cuda()
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')




@HOOKS.register_module()
class EfficientTeacherHook(Hook):
    def __init__(self):
        self.ema_cfg = dict(
            ema_type='ExpMomentumEMA',
            momentum=0.0002,
            update_buffers=True,
        )
        self.copy_para = False

    def before_run(self, runner: Runner) -> None:
        """To check that teacher model and student model exist."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if hasattr(model, 'teacher'):
            assert hasattr(model, 'teacher')
            assert hasattr(model, 'student')
            assert model.teacher == None
            self.src_model = model.student

            ema_type = self.ema_cfg.pop('ema_type')
            self.ema_cfg['type'] = ema_type

            self.ema_model = MODELS.build(self.ema_cfg, default_args=dict(model=self.src_model)).eval()
            print('set teacher no grad')
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            model.teacher = self.ema_model
            model.init_teacher = True
        else:
            assert isinstance(model, DetTTAModel)
            self.src_model = model.module.student

            ema_type = self.ema_cfg.pop('ema_type')
            self.ema_cfg['type'] = ema_type

            self.ema_model = MODELS.build(self.ema_cfg, default_args=dict(model=self.src_model)).eval()
            print('set teacher no grad')
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            model.module.teacher = self.ema_model
            model.module.init_teacher = True


    def before_train(self, runner) -> None:
        if (self.copy_para is False) and (not runner._resume):
            ema_params = self.ema_model.module.state_dict()
            src_params = self.src_model.state_dict()
            for k, p in ema_params.items():
                p.data.copy_(src_params[k].data)
            self.copy_para = True
            print('确保万无一失！再copy一次模型参数')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:

        self.ema_model.update_parameters(self.src_model)
        # 由于model是直接定义在efficientteacher类里的，所以不需要保存时候特殊处理


@MODELS.register_module()
class SemiDataPreprocessor(MultiBranchDataPreprocessor):
    def __init__(self):
        super().__init__(data_preprocessor=dict(
            type='mmdet.DetDataPreprocessor',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False,
            batch_augments=None))

        # mmdet_cfg = dict(
        #     type='mmdet.DetDataPreprocessor',
        #     mean=[103.53, 116.28, 123.675],
        #     std=[57.375, 57.12, 58.395],
        #     bgr_to_rgb=False,
        #     batch_augments=None)
        mmyolo_cfg = dict(
            type='YOLOv5DetDataPreprocessor',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False)

        # self.mmdet_datapreprocessor = MODELS.build(mmdet_cfg)
        self.mmyolo_datapreprocessor = MODELS.build(mmyolo_cfg)
        self.mmyolo_datapreprocessor.cuda()
        self.cuda()

    def forward(self, data: dict, training: bool = False):
        if isinstance(data['data_samples'], dict) and ('sup' in data['data_samples']):
            # print('11111')
            # data['inputs']['sup'][0] = data['inputs']['sup'][0].cuda()
            # data['inputs']['unsup_teacher'][1] = data['inputs']['unsup_teacher'][1].cuda()
            # data['inputs']['unsup_student'][1] = data['inputs']['unsup_student'][1].cuda()
            return super().forward(data, training)
        else:
            return self.mmyolo_datapreprocessor(data, training)
