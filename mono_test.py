# -*- coding: utf-8 -*-

import sys
import os
import shutil
import time
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import argparse
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from boxx import g, show

from models.bts import *
from models.FusionNet import *
from mono_dataset import MonoDataset
from mono_utils.mono_utils import *


def test_net(model, dataloader_val, device, mea_nums):
    """用验证集评判网络性能"""
    model.eval()
    metrics_test = Record(mea_nums)
    time_cost_ = Record()
    with torch.no_grad():
        for cnt, (img, lidar_in, depth_gt) in enumerate(tqdm(dataloader_val), 1):
            img = img.to(device, non_blocking=True)
            lidar_in = lidar_in.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)

            torch.cuda.synchronize()
            time_a = time.perf_counter()

            # depth_est, _, _, _, _ = model(img)

            input = torch.cat((lidar_in, img), 1)
            output = model(input)
            depth_est, _, _, _ = output[0], output[1], output[2], output[3]

            depth_est = torch.clamp(depth_est, 0, args.max_depth)
            torch.cuda.synchronize()
            time_use_ = time.perf_counter() - time_a

            valid_mask = get_valid_mask(depth_gt, args)
            # metrics_tmp, min_is_best_mask, metrics_names = compute_errors_9(depth_gt, depth_est, valid_mask)
            metrics_tmp, min_is_best_mask, metrics_names = compute_errors_3(depth_gt, depth_est, valid_mask)

            metrics_test.update(metrics_tmp, img.shape[0])
            time_cost_.update(time_use_, img.shape[0])

            depth_est = (depth_est * 100).detach().cpu().numpy().squeeze().astype(np.uint16).squeeze()
            depth_gt = (depth_gt * 100).detach().cpu().numpy().squeeze().astype(np.uint16).squeeze()
            lidar_in = (lidar_in * 100).detach().cpu().numpy().squeeze().astype(np.uint16)
            torchvision.utils.save_image(img, os.path.join(args.save_dir, 'rgb', 'rgb-{}.jpg'.format(cnt)),
                                         normalize=True, scale_each=True)
            cv2.imwrite(os.path.join(args.save_dir, 'depth-est', 'depth-est-{}.png'.format(cnt)), depth_est)
            cv2.imwrite(os.path.join(args.save_dir, 'depth-gt', 'depth-gt-{}.png'.format(cnt)), depth_gt)
            cv2.imwrite(os.path.join(args.save_dir, 'lidar', 'lidar-{}.png'.format(cnt)), lidar_in)

    return metrics_test.avg, min_is_best_mask, metrics_names, time_cost_.avg


if __name__ == '__main__':
    __spec__ = None

    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--data_path', type=str, help='path to the data', required=True)
    parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu or ob', default='nyu')
    parser.add_argument('--filelist_txt_test', type=str, help='path to the filenames text file', required=True)
    parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--do_eigen_crop', help='if do eigen crop to test', action='store_true')
    parser.add_argument('--sparse_ratio', type=float, help='sparse_ratio', default=0.01)

    parser.add_argument('--gpu', type=int, help='GPU id to use.', default=None)
    parser.add_argument('--variance_focus', type=float, help='lambda in paper: [0, 1]', default=0.85)
    parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
    parser.add_argument('--save_dir', type=str, help='directory to save result', default='')
    parser.add_argument('--msg', type=str, help='some message', default='')
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--encoder', type=str, help='type of encoder, desenet121_bts, densenet161_bts, resnet101_bts, '
                                                    'resnet50_bts, resnext50_bts or resnext101_bts',
                        default='densenet121_bts')
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # 加载数据
    dataset_test = MonoDataset(args.data_path, args.filelist_txt_test, mode=args.mode, sparse_ratio=args.sparse_ratio,
                               max_depth=args.max_depth, dataset_name=args.dataset, mul_times=32)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False,
                                 num_workers=args.num_threads, pin_memory=True)

    # 定义网络等
    device_str = "cpu" if args.gpu is None else "cuda:{}".format(args.gpu)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    # model = BtsModel(args.max_depth, args.encoder).to(device)
    # set_misc(model)
    # mea_nums = 9

    model = uncertainty_net(in_channels=4).to(device)
    mea_nums = 3

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # 初始化
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'depth-est'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'depth-gt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'lidar'), exist_ok=True)

    # 预测
    print('Start Testing...')
    eval_measures, min_is_best_mask, measures_names, time_use_avg = test_net(model, dataloader_test, device, mea_nums)
    torch.cuda.empty_cache()
    msg = get_measures_msg(eval_measures, measures_names)
    print(msg)
    print(time_use_avg[0])
