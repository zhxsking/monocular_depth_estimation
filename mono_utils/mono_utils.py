# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import torch
from torch.optim import lr_scheduler
import torch.nn.init as init


def linear_p_full(img, p=0.02):
    """对图像进行linear p 处理"""

    p *= 100
    res = np.zeros(img.shape).astype(np.float32)

    high_value = np.nanpercentile(img, 100-p)  # 取得98%直方图处对应灰度
    low_value = np.nanpercentile(img, p)  # 取得2%直方图处对应灰度
    tmp = np.clip(img, a_min=low_value, a_max=high_value)
    tmp = ((tmp - low_value) / (high_value - low_value + 1e-5))

    return tmp


def linear_p(img, p=0.02):
    """对图像进行linear p 处理"""
    dim = 1
    if img.ndim > 2:  # 判断是否为灰度图
        dim = img.shape[2]

    p *= 100
    res = np.zeros(img.shape).astype(np.float32)
    for i in range(dim):
        if img.ndim > 2:
            tmp = img[:, :, i]
        else:
            tmp = img

        high_value = np.nanpercentile(tmp, 100-p)  # 取得98%直方图处对应灰度
        low_value = np.nanpercentile(tmp, p)  # 取得2%直方图处对应灰度
        tmp = np.clip(tmp, a_min=low_value, a_max=high_value)
        tmp = ((tmp - low_value) / (high_value - low_value + 1e-5))

        if img.ndim > 2:
            res[:, :, i] = tmp
        else:
            res = tmp
    return res


def linear_p_tensor(tensor, p=0.02):
    res = np.zeros(tensor.shape, dtype=np.float32)
    tmp = tensor.clone()
    for i in range(res.shape[0]):
        res[i, :] = linear_p_full(tmp[i, :].detach().cpu().numpy(), p)
    res = torch.from_numpy(res)
    return res


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, lr_policy, gamma=0.5, step_size=5, end_lr=1e-5, step_all=100):
    if lr_policy == 'lambda':
        def lambda_rule(epoch, num_epochs=step_all, power=0.9):
            return (1 - epoch / num_epochs) ** power
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma,
                                                   min_lr=end_lr, patience=step_size)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def read_depth(path, dataset, max_depth):
    """读取深度图并化成米单位"""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if dataset == 'kitti':
        depth /= 256
    elif dataset == 'nyu':
        depth /= 1000
    elif dataset == 'ob':
        depth /= 100
    elif dataset == 'irs':
        depth = depth[:, :, 1]  # irs是视差图
        depth[depth < 1] = 1
        depth = 48 / depth  # baseline 0.1m, f 480 pixels
    else:
        depth /= 100
    depth[depth > max_depth] = max_depth
    return depth


def get_measures_msg(measures, measures_names):
    msg = ''
    for i in range(len(measures)):
        msg += '{}: {:.5f}, '.format(measures_names[i], measures[i])
    return msg


def get_valid_mask(depth_gt_, args):
    valid_mask = (depth_gt_ > 0) & (depth_gt_ < args.max_depth)

    gt_height, gt_width = depth_gt_.shape[2:]
    eval_mask = torch.zeros(valid_mask.shape)
    if args.do_eigen_crop:
        if args.dataset == 'kitti':
            eval_mask[:, :, int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        elif args.dataset == 'nyu':
            eval_mask[:, :, 45:471, 41:601] = 1
        valid_mask = torch.logical_and(valid_mask, eval_mask)

    return valid_mask


def compute_errors_9_np(gt, pred, mask_=None):
    """计算评价指标，务必保证第一个指标为常用且越小越好的指标，以保证学习率衰减合法"""
    if mask_ is not None:
        gt = gt[mask_]
        pred = pred[mask_]
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    errors = np.array([abs_rel, sq_rel, rmse, log_rms, log10, silog, d1, d2, d3], dtype=np.float32)
    min_is_best_mask_ = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    errors_names = ['abs_rel', 'sq_rel', 'rmse', 'log_rms', 'log10', 'silog', 'd1', 'd2', 'd3']
    return errors, min_is_best_mask_, errors_names


def compute_errors_9(gt, pred, mask_=None):
    """计算评价指标，务必保证第一个指标为常用且越小越好的指标，以保证学习率衰减合法"""
    if mask_ is not None:
        gt = gt[mask_]
        pred = pred[mask_]
    thresh = torch.max((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_diff = (pred - gt).abs()
    err_log = torch.log(gt) - torch.log(pred)

    rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
    log_rms = torch.sqrt(torch.mean(torch.pow(err_log, 2))).item()
    abs_rel = (abs_diff / gt).mean().item()
    sq_rel = ((torch.pow(abs_diff, 2)) / gt).mean().item()
    silog = torch.sqrt((torch.pow(err_log, 2)).mean() - torch.pow(err_log.mean(), 2)).item() * 100
    log10 = err_log.abs().mean().item()

    errors = np.array([abs_rel, sq_rel, rmse, log_rms, log10, silog, d1, d2, d3], dtype=np.float32)
    min_is_best_mask_ = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
    errors_names = ['abs_rel', 'sq_rel', 'rmse', 'log_rms', 'log10', 'silog', 'd1', 'd2', 'd3']
    return errors, min_is_best_mask_, errors_names


def compute_errors_3(gt, pred, mask_=None):
    """计算评价指标，务必保证第一个指标为常用且越小越好的指标，以保证学习率衰减合法"""
    if mask_ is not None:
        gt = gt[mask_]
        pred = pred[mask_]

    abs_diff = (pred - gt).abs()
    rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
    mae = abs_diff.mean().item()
    abs_rel = (abs_diff / gt).mean().item()

    errors = np.array([rmse, mae, abs_rel], dtype=np.float32)
    min_is_best_mask_ = np.array([1, 1, 1], dtype=np.uint8)
    errors_names = ['rmse', 'mae', 'abs_rel']
    return errors, min_is_best_mask_, errors_names


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


class Record(object):
    """记录loss、acc等信息"""
    def __init__(self, metric_nums=1):
        self.val = np.zeros(metric_nums, dtype=np.float32)
        self.avg = np.zeros(metric_nums, dtype=np.float32)
        self.sum = np.zeros(metric_nums, dtype=np.float32)
        self.cnt = 0.0
        self.avg_list = []
        self.metric_nums = metric_nums

    def reset(self):
        self.val = np.zeros(self.metric_nums, dtype=np.float32)
        self.avg = np.zeros(self.metric_nums, dtype=np.float32)
        self.sum = np.zeros(self.metric_nums, dtype=np.float32)
        self.cnt = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def save_avg_list(self):
        self.avg_list.append(self.avg)


class Logger(object):
    """保存日志信息"""
    def __init__(self):
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'a'
        self.file = open(file, mode)
        self.file.write('\n--------------------{}--------------------\n'
                        .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.file.flush()

    def write(self, msg, do_print=False):
        if do_print:
            print(msg)
        self.file.write(msg)
        self.file.write('\n')
        self.file.flush()
    
    def close(self):
        self.file.write('---------------------------------------------\n')
        self.file.close()


if __name__ == '__main__':
    import time
    from PIL import Image
    import matplotlib.pyplot as plt

    # path = r"C:\pic\ob_dataset\sync\baseline_large\outdoor\FactoryDistrict\1RgbLeftLarge.jpg"
    path = r"D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\groundtruth_depth\ob-1.png"
    # img = plt.imread(path)
    img = np.array(Image.open(path), dtype=np.uint16)

    since = time.time()
    res1 = linear_p_1(img, 0.02)
    stop = time.time()
    print(stop - since)

    since = time.time()
    res2 = linear_p(img, 0.02)
    stop = time.time()
    print(stop - since)

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(res1)
    plt.figure()
    plt.imshow(res2)
