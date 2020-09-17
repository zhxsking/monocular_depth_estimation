# -*- coding: utf-8 -*-

import os
import re
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def gen_filelist_txt(dir_path_, txt_path_):
    """获取数据集文件路径列表，并将其保存到txt"""
    rgb_list_, depth_list_, lidar_list_ = [], [], []
    filelist_ = []
    for home, dirs, files in os.walk(dir_path_):
        for filename in files:
            abs_path = os.path.join(home, filename)  # 文件绝对路径
            rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
            rel_path = rel_path.replace('\\', '/')

            if 'rgb' in rel_path:
                rgb_list_.append(rel_path)
            if 'depth' in rel_path:
                depth_list_.append(rel_path)
            if 'lidar' in rel_path:
                lidar_list_.append(rel_path)

    for i in range(len(rgb_list_)):
        tmp = rgb_list_[i] + ' '
        tmp += depth_list_[i] if len(depth_list_) != 0 else 'none'
        tmp += ' '
        tmp += lidar_list_[i] if len(lidar_list_) != 0 else 'none'
        tmp += ' '
        filelist_.append(tmp)

    # 文件列表保存为txt
    f = open(txt_path_, "w+")
    for item in filelist_:
        f.write(item + "\n")
    f.close()
    print(len(filelist_))


def gen_filelist_txt_ob_dataset(dir_path_, txt_path_, seed=1):
    """获取ob数据集文件路径列表，并将其保存到txt，仅用左图版"""
    filelist_ = []
    rgb_left_list = []
    rgb_right_list = []
    depth_left_list = []
    for home, dirs, files in os.walk(dir_path_):
        for filename in files:
            abs_path = os.path.join(home, filename)  # 文件绝对路径
            rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
            rel_path = rel_path.replace('\\', '/')

            if 'RgbLeft' in rel_path:
                rgb_left_list.append(rel_path)
            if 'RgbRight' in rel_path:
                rgb_right_list.append(rel_path)
            if 'DepthLeft' in rel_path:
                depth_left_list.append(rel_path)

    rgb_left_list = sorted(rgb_left_list)
    # rgb_right_list = sorted(rgb_right_list)
    depth_left_list = sorted(depth_left_list)

    for i in range(len(rgb_left_list)):
        tmp = rgb_left_list[i] + ' ' + depth_left_list[i] + ' 100'
        filelist_.append(tmp)

    # 切分为训练集和验证集
    train_files, val_files = train_test_split(filelist_, test_size=0.05, random_state=seed)

    # 训练集文件列表保存为txt
    f = open(txt_path_ + r"-train", "w+")
    for item in train_files:
        f.write(item + "\n")
    f.close()

    # 验证集文件列表保存为txt
    f = open(txt_path_ + r"-val", "w+")
    for item in val_files:
        f.write(item + "\n")
    f.close()

    print(len(filelist_), len(train_files), len(val_files))


def gen_filelist_txt_irs_dataset(dir_path_, txt_path_, seed=1):
    """获取IRS数据集文件路径列表，并将其保存到txt"""
    filelist_ = []
    rgb_left_list = []
    rgb_right_list = []
    depth_left_list = []
    for home, dirs, files in os.walk(dir_path_):
        for filename in files:
            abs_path = os.path.join(home, filename)  # 文件绝对路径
            rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
            rel_path = rel_path.replace('\\', '/')

            if re.search('/l_', rel_path):
                rgb_left_list.append(rel_path)
            if re.search('/r_', rel_path):
                rgb_right_list.append(rel_path)
            if re.search('/d_', rel_path):
                depth_left_list.append(rel_path)

    rgb_left_list = sorted(rgb_left_list)
    # rgb_right_list = sorted(rgb_right_list)
    depth_left_list = sorted(depth_left_list)

    for i in range(len(rgb_left_list)):
        tmp = rgb_left_list[i] + ' ' + depth_left_list[i] + ' none'
        filelist_.append(tmp)

    # 切分为训练集和验证集
    train_files, val_files = train_test_split(filelist_, test_size=0.05, random_state=seed)

    # 训练集文件列表保存为txt
    f = open(txt_path_ + r"-train", "w+")
    for item in train_files:
        f.write(item + "\n")
    f.close()

    # 验证集文件列表保存为txt
    f = open(txt_path_ + r"-val", "w+")
    for item in val_files:
        f.write(item + "\n")
    f.close()

    print(len(filelist_), len(train_files), len(val_files))


def gen_filelist_txt_kitti_dataset(dir_path_, txt_path_, do_shuffle=True):
    """获取kitti数据集文件路径列表，并将其保存到txt"""
    cnt_all = 0
    for _, _, _ in os.walk(dir_path_):
        cnt_all += 1

    rgb_list_train, rgb_list_val, rgb_list_test = [], [], []
    lidar_list_train, lidar_list_val, lidar_list_test = [], [], []
    depth_list_train, depth_list_val, depth_list_test = [], [], []
    pbar = tqdm(total=cnt_all, desc='Search images')
    for home, dirs, files in os.walk(dir_path_):
        pbar.update(1)
        if re.search('train', home):
            if re.search('rgb', home) and re.search('image_02', home):  # or re.search('image_03', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    rgb_list_train.append(rel_path)
            if re.search('proj_depth', home) and re.search('velodyne_raw', home) and re.search('image_02', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    lidar_list_train.append(rel_path)
            if re.search('proj_depth', home) and re.search('groundtruth', home) and re.search('image_02', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    depth_list_train.append(rel_path)

        if re.search('val', home):
            if re.search('rgb', home) and re.search('image_02', home):  # or re.search('image_03', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    rgb_list_val.append(rel_path)
            if re.search('proj_depth', home) and re.search('velodyne_raw', home) and re.search('image_02', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    lidar_list_val.append(rel_path)
            if re.search('proj_depth', home) and re.search('groundtruth', home) and re.search('image_02', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    depth_list_val.append(rel_path)
        if re.search('val_selection_cropped', home):
            if re.search('image', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    rgb_list_test.append(rel_path)
            if re.search('velodyne_raw', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    lidar_list_test.append(rel_path)
            if re.search('groundtruth', home):
                for filename in files:
                    abs_path = os.path.join(home, filename)  # 文件绝对路径
                    rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
                    rel_path = rel_path.replace('\\', '/')
                    depth_list_test.append(rel_path)
    pbar.close()
    rgb_list_train, rgb_list_val, rgb_list_test = sorted(rgb_list_train), sorted(rgb_list_val), sorted(rgb_list_test)
    lidar_list_train, lidar_list_val, lidar_list_test = sorted(lidar_list_train), sorted(lidar_list_val), sorted(lidar_list_test)
    depth_list_train, depth_list_val, depth_list_test = sorted(depth_list_train), sorted(depth_list_val), sorted(depth_list_test)

    # 训练集文件列表合并，方便shuffle
    train_files = []
    for i in range(len(rgb_list_train)):
        tmp = '{} {} {}'.format(rgb_list_train[i], depth_list_train[i], lidar_list_train[i])
        train_files.append(tmp)
    if do_shuffle:
        random.shuffle(train_files)

    # 训练集文件列表保存为txt
    f = open(txt_path_ + r"-train", "w+")
    for item in train_files:
        f.write(item + "\n")
    f.close()

    # 验证集文件列表保存为txt
    f = open(txt_path_ + r"-val", "w+")
    for i in range(len(rgb_list_val)):
        item = '{} {} {}\n'.format(rgb_list_val[i], depth_list_val[i], lidar_list_val[i])
        f.write(item)
    f.close()

    # 测试集文件列表保存为txt
    f = open(txt_path_ + r"-test", "w+")
    for i in range(len(rgb_list_test)):
        item = '{} {} {}\n'.format(rgb_list_test[i], depth_list_test[i], lidar_list_test[i])
        f.write(item)
    f.close()

    print(len(rgb_list_train), len(rgb_list_val), len(rgb_list_test))


def gen_filelist_txt_nyu_dataset(dir_path_, txt_path_, do_shuffle=True):
    """获取NYU数据集文件路径列表，并将其保存到txt"""
    rgb_list, rgb_list_test = [], []
    depth_list, depth_list_test = [], []
    for home, dirs, files in os.walk(dir_path_):
        for filename in files:
            abs_path = os.path.join(home, filename)  # 文件绝对路径
            rel_path = os.path.relpath(abs_path, dir_path_)  # 文件相对路径
            rel_path = rel_path.replace('\\', '/')

            if re.search('BTS_data', rel_path) and re.search('rgb', rel_path):
                rgb_list.append(rel_path)
            if re.search('BTS_data', rel_path) and re.search('sync_depth', rel_path):
                depth_list.append(rel_path)
            if re.search('official_splits/test', rel_path) and re.search('rgb', rel_path):
                rgb_list_test.append(rel_path)
            if re.search('official_splits/test', rel_path) and re.search('sync_depth', rel_path):
                depth_list_test.append(rel_path)

    rgb_list, rgb_list_test = sorted(rgb_list), sorted(rgb_list_test)
    depth_list, depth_list_test = sorted(depth_list), sorted(depth_list_test)

    # 训练集文件列表合并，方便shuffle
    train_files = []
    for i in range(len(rgb_list)):
        tmp = '{} {} none'.format(rgb_list[i], depth_list[i])
        train_files.append(tmp)
    if do_shuffle:
        random.shuffle(train_files)

    # 训练集文件列表保存为txt
    f = open(txt_path_ + r"-train", "w+")
    for item in train_files:
        f.write(item + "\n")
    f.close()

    # 测试集文件列表保存为txt
    f = open(txt_path_ + r"-test", "w+")
    for i in range(len(rgb_list_test)):
        item = '{} {} none\n'.format(rgb_list_test[i], depth_list_test[i])
        f.write(item)
    f.close()

    print(len(rgb_list), len(rgb_list_test))


def gen_filelist_txt_from_all_path(txt_path_from_, txt_path_, subset='irs', seed=1):
    """获取NYU数据集文件路径列表，并将其保存到txt"""

    all_path_list = list(pd.read_csv(txt_path_from_, header=None)[0])
    subset_list = []

    if subset == 'irs':
        subset_list = all_path_list[216120:]
    elif subset == 'd435':
        subset_list = all_path_list[:46096]

    # 切分为训练集和验证集
    train_files, val_files = train_test_split(subset_list, test_size=0.05, random_state=seed)

    # 训练集文件列表保存为txt
    f = open(txt_path_ + r"-train", "w+")
    for item in train_files:
        f.write(item + "\n")
    f.close()

    # 验证集文件列表保存为txt
    f = open(txt_path_ + r"-val", "w+")
    for item in val_files:
        f.write(item + "\n")
    f.close()

    print(len(subset_list), len(train_files), len(val_files))


if __name__ == "__main__":
    # dir_path = r"D:\code\python\monocular_depth_estimation\test_pics"
    # txt_path = r"../filelist-bts-test"
    # gen_filelist_txt(dir_path, txt_path)

    # dir_path = r"C:\pic\ob_dataset\sync\baseline_large"
    # txt_path = r"../filelist-ob-dataset"
    # filelist = gen_filelist_txt_ob_dataset(dir_path, txt_path)
    # print(len(filelist))

    # dir_path = r"D:\pic\IRS"
    # # dir_path = r"E:\data\home\xubin\dataset\dataset\IRSDataset"
    # txt_path = r"../filelist-irs-s-dataset"
    # gen_filelist_txt_irs_dataset(dir_path, txt_path)

    dir_path = r"D:\pic\KITTI\sync"
    txt_path = r"../filelist-kitti-dataset"
    gen_filelist_txt_kitti_dataset(dir_path, txt_path)

    # dir_path = r"C:\pic\NYU-depth"
    # txt_path = r"../filelist-nyu-dataset"
    # gen_filelist_txt_nyu_dataset(dir_path, txt_path)

    # txt_path_from = r"stage3_d435speckle.txt"
    # txt_path = r"../filelist-lmdb-subset-irs"
    # gen_filelist_txt_from_all_path(txt_path_from, txt_path)
