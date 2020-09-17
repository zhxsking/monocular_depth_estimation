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
import torchvision.transforms as transforms
from imgaug import augmenters as iaa

from models.bts import *
from models.FusionNet import *
from mono_dataset import MonoDataset
from mono_utils.mono_utils import *


if __name__ == '__main__':
    __spec__ = None

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # dataset = 'irs'
    # p_img = r'D:\pic\IRS\Home\ModernHotelRoom\l_1.png'
    # p_lidar = r''
    # p_depth = r'D:\pic\IRS\Home\ModernHotelRoom\d_1.exr'
    # p_ck = r'checkpoint\model-13500-best-rmse-1.20500'

    dataset = 'kitti'
    p_img = r'D:\pic\KITTI\sync\data_depth\val_selection_cropped\image\2011_09_26_drive_0002_sync_image_0000000005_image_02.png'
    p_lidar = r'D:\pic\KITTI\sync\data_depth\val_selection_cropped\velodyne_raw\2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png'
    p_depth = r'D:\pic\KITTI\sync\data_depth\val_selection_cropped\groundtruth_depth\2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png'
    p_ck = r'checkpoint\model-13500-best-rmse-1.20500'

    # -----------------------------------------------------------------------------------------------------------------#
    parser.add_argument('--path_img', type=str, help='path to the data', default=p_img)
    parser.add_argument('--path_lidar', type=str, help='path to the data', default=p_lidar)
    parser.add_argument('--path_depth', type=str, help='path to the data', default=p_depth)
    parser.add_argument('--checkpoint_path', type=str, help='path of checkpoint to load', default=p_ck)
    parser.add_argument('--save_dir', type=str, help='directory to save result', default='result')
    # -----------------------------------------------------------------------------------------------------------------#

    parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu or ob or irs', default=dataset)
    parser.add_argument('--do_eigen_crop', help='if do eigen crop to test', action='store_true')
    parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=85)
    parser.add_argument('--sparse_ratio', type=float, help='sparse_ratio', default=0.01)
    parser.add_argument('--gpu', type=int, help='GPU id to use.', default=0)
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--encoder', type=str, help='type of encoder, desenet121_bts, densenet161_bts, resnet101_bts, '
                                                    'resnet50_bts, resnext50_bts or resnext101_bts',
                        default='densenet121_bts')
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    img_process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    to_tensor = transforms.ToTensor()
    to_sparse = iaa.Sequential([iaa.Dropout(1 - args.sparse_ratio), ], random_order=True)

    # 加载数据
    img = Image.open(args.path_img).convert("RGB")
    img = np.asarray(img, dtype=np.float32) / 255.0
    lidar = None
    depth = None
    if args.path_depth != '':
        # depth = cv2.imread(args.path_depth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100
        depth = read_depth(args.path_depth, args.dataset, args.max_depth)
    else:
        depth = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    if args.path_lidar != '':
        # lidar = cv2.imread(args.path_lidar, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100
        lidar = read_depth(args.path_lidar, args.dataset, args.max_depth)
    else:
        lidar = to_sparse(image=depth)


    # pad为网络可输入的大小
    mul_times = 32
    h_ori, w_ori = img.shape[0], img.shape[1]
    h_pad = int(np.ceil(h_ori / mul_times) * mul_times)
    w_pad = int(np.ceil(w_ori / mul_times) * mul_times)
    img = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=img)
    lidar = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=lidar)
    depth = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=depth)

    # 标准化
    img = img_process(img.copy()).unsqueeze(0)
    depth = to_tensor(depth.copy()).unsqueeze(0)
    lidar = to_tensor(lidar.copy()).unsqueeze(0)

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
    model.load_state_dict(checkpoint['model'], False)

    # 初始化
    os.makedirs(args.save_dir, exist_ok=True)
    print('Start Testing...')
    model.eval()
    with torch.no_grad():
        img_in = img.to(device)
        lidar_in = lidar.to(device)
        depth_gt = depth.to(device)

        torch.cuda.synchronize()
        time_a = time.perf_counter()

        # depth_est, _, _, _, _ = model(img_in)

        input = torch.cat((lidar, img), 1).to(device)
        output = model(input)
        depth_est, _, _, _ = output[0], output[1], output[2], output[3]

        depth_est = torch.clamp(depth_est, 0, args.max_depth)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time_use = time.perf_counter() - time_a

        valid_mask = get_valid_mask(depth_gt, args)
        metrics_tmp, min_is_best_mask, metrics_names = compute_errors_9(depth_gt, depth_est, valid_mask)
        # metrics_tmp, min_is_best_mask, metrics_names = compute_errors_3(depth_gt, depth_est, valid_mask)

        print(time_use)
        show(depth_est)

        depth_est = (depth_est * 100).detach().cpu().numpy().squeeze().astype(np.uint16).squeeze()
        depth_gt = (depth_gt * 100).detach().cpu().numpy().squeeze().astype(np.uint16).squeeze()
        lidar_in = (lidar_in * 100).detach().cpu().numpy().squeeze().astype(np.uint16)
        torchvision.utils.save_image(img_in, os.path.join(args.save_dir, 'rgb.jpg'), normalize=True, scale_each=True)
        cv2.imwrite(os.path.join(args.save_dir, 'depth-est.png'), depth_est)
        cv2.imwrite(os.path.join(args.save_dir, 'depth-gt.png'), depth_gt)
        cv2.imwrite(os.path.join(args.save_dir, 'lidar.png'), lidar_in)
    torch.cuda.empty_cache()
    msg = get_measures_msg(metrics_tmp, metrics_names)
    print(msg)
