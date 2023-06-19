import numpy as np

from mmyolo.datasets import YOLOv5KeepRatioResize, LetterResize

transform = [
            YOLOv5KeepRatioResize(scale=(32, 32)),
            LetterResize(scale=(64, 68), pad_val=dict(img=144))
        ]

# input_h, input_w, output_h, output_w = 247, 314, 150, 151
# input_h, input_w, output_h, output_w = 385, 662, 532, 138
input_h, input_w, output_h, output_w = 596, 235, 315, 515
data_info = dict(
    img=np.random.random((input_h, input_w, 3)),
    gt_bboxes=np.array([[0, 0, 5, 5]], dtype=np.float32),
    batch_shape=np.array([output_h, output_w], dtype=np.int64))
for t in transform:
    data_info = t(data_info)
    print(data_info['img'].shape, data_info['scale_factor'])
# print(data_info['img'].shape)
# print(data_info['scale_factor'])

scale_factor = np.asarray(
                data_info['scale_factor'])[::-1]  # (w, h) -> (h, w)

max_long_edge = max((32, 32))
max_short_edge = min((32, 32))
scale_factor_keepratio = min(max_long_edge / max(input_h, input_w),
                             max_short_edge / min(input_h, input_w))
validate_shape = np.asarray((int(input_h*scale_factor_keepratio), int(input_w*scale_factor_keepratio)))
print('validate_shape111', validate_shape)
scale_factor_keepratio = np.asarray((validate_shape[1]/input_w, validate_shape[0]/input_h))
print('scale_factor_keepratio111', scale_factor_keepratio)


pad_param = data_info['pad_param'].reshape(-1, 2).sum(1)    # (top, b, l, r) -> (h, w)
scale_factor_letter = ((np.asarray((output_h, output_w)) - pad_param) / validate_shape)[::-1]
print('scale_factor_letter', scale_factor_letter)

print(scale_factor_keepratio * scale_factor_letter[::-1], scale_factor)
print([output_h, output_w], data_info['img'].shape[:2])

# ratio = min(output_h / validate_shape[0], output_w / validate_shape[1])
# ratio = [ratio, ratio]
# no_pad_shape = (int(round(validate_shape[0]*ratio[0])),
#                 int(round(validate_shape[1]*ratio[1])))
# scale_factor_letter = ()
#
#
# scale_factor_letter = np.asarray((output_h, output_w)) / np.asarray(validate_shape)
#
# pad_param = data_info['pad_param'].reshape(-1, 2).sum(
#                 1)  # (top, b, l, r) -> (h, w)
# scale_factor_letter = (
#     scale_factor_letter -
#     (pad_param / validate_shape))[np.argmin(scale_factor_letter)]
# print('scale_factor_letter', scale_factor_letter)
# print(scale_factor == (scale_factor_keepratio *
#                                               scale_factor_letter))