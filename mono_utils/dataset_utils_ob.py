# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import multiprocessing


def analyse_ob_dataset(dir_path_):
    """分析数据集的直方图"""
    hist_all = np.zeros((65536, 1), dtype=np.float32)
    cnt = 0
    for home, dirs, files in os.walk(dir_path_):
        for filename in files:
            if 'DepthLeft' in filename:
                cnt += 1

    pbar = tqdm(total=cnt, desc='caling')
    for p_dir in os.listdir(dir_path_):
        hist_ = np.zeros((65536, 1), dtype=np.float32)
        for home, dirs, files in os.walk(os.path.join(dir_path_, p_dir)):
            for filename in files:
                abs_path = os.path.join(home, filename)  # 文件绝对路径
                if 'DepthLeft' in filename:
                    pbar.update(1)
                    im = cv2.imread(abs_path, cv2.IMREAD_ANYDEPTH)
                    hist_tmp = cv2.calcHist([im], [0], None, [65536], [0, 65536])
                    hist_tmp[60000:, :] = 0
                    hist_ += hist_tmp
                    hist_all += hist_tmp
        plt.figure()
        plt.plot(hist_)
        plt.title(p_dir)
        plt.xlabel('depth(cm)')
        save_path = (p_dir + '-hist.jpg')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.figure()
    plt.plot(hist_all)
    plt.title('all data')
    plt.xlabel('depth(cm)')
    save_path = ('all-data-hist.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pbar.close()
    return hist_all


def gen_filelist_txt_ob_dataset(dir_path_, txt_path_, seed=1):
    """获取数据集文件路径列表，并将其保存到txt"""
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
    rgb_right_list = sorted(rgb_right_list)
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

    return filelist_


def proc_depth_img(depth_img, max_rela_depth=10000):
    depth_img[depth_img > max_rela_depth] = max_rela_depth
    return depth_img


def save_img(filename, dir_in_, dir_out_, abs_path, q=None):
    """将数据按小基线、大基线分成两组并转换文件格式"""
    path_baseline_small = os.path.join(dir_out_, 'baseline_small')
    path_baseline_large = os.path.join(dir_out_, 'baseline_large')
    os.makedirs(path_baseline_small, exist_ok=True)
    os.makedirs(path_baseline_large, exist_ok=True)

    out_path = ''
    img = None
    if 'Large' in filename:
        out_dir_ = os.path.join(path_baseline_large, abs_path.split(dir_in_)[1].split(filename)[0][1:])
        os.makedirs(out_dir_, exist_ok=True)
        if 'Rgb' in filename:
            out_path = os.path.join(out_dir_, filename.split('.png')[0] + '.jpg')
            img = cv2.imread(abs_path)
        if 'Depth' in filename:
            # out_path = os.path.join(out_dir_, filename.split('.hdr')[0] + '.png')
            # img = cv2.imread(abs_path, cv2.IMREAD_ANYDEPTH)[:, :, 2].astype(np.uint16)
            out_path = os.path.join(out_dir_, filename.split('.hdr')[0] + '.png')
            img = cv2.imread(abs_path, cv2.IMREAD_ANYDEPTH)[:, :, 2].astype(np.uint16)
            # img = proc_depth_img(img, max_rela_depth=10000)
    else:
        out_dir_ = os.path.join(path_baseline_small, abs_path.split(dir_in_)[1].split(filename)[0][1:])
        os.makedirs(out_dir_, exist_ok=True)
        if 'Rgb' in filename:
            out_path = os.path.join(out_dir_, filename.split('.png')[0] + '.jpg')
            img = cv2.imread(abs_path)
        if 'Depth' in filename:
            out_path = os.path.join(out_dir_, filename.split('.hdr')[0] + '.png')
            img = cv2.imread(abs_path, cv2.IMREAD_ANYDEPTH)[:, :, 2].astype(np.uint16)
            # img = proc_depth_img(img, max_rela_depth=10000)
    if not(q is None):
        q.put(filename)
    msg = cv2.imwrite(out_path, img)
    del img
    return msg


def split_ob_dataset(dir_in_, dir_out_):
    """切分OB数据集并保存到相应文件夹"""
    cnt_success = 0
    cnt_all = 0
    for home, dirs, files in os.walk(dir_in_):
        cnt_all += len(files)

    pbar = tqdm(total=cnt_all, desc='Convert image')
    for home, dirs, files in os.walk(dir_in_):
        for filename in files:
            pbar.update(1)
            abs_path = os.path.join(home, filename)  # 文件绝对路径

            if save_img(filename, dir_in_, dir_out_, abs_path):
                 cnt_success += 1
    pbar.close()
    print('success/all: {}/{}'.format(cnt_success, cnt_all))


def split_ob_dataset_threads(dir_in_, dir_out_, num_threads=4):
    """切分OB数据集并保存到相应文件夹，内存会不断增加"""
    cnt_all = 0
    for home, dirs, files in os.walk(dir_in_):
        cnt_all += len(files)

    print('Running...')
    p = multiprocessing.Pool(processes=num_threads, maxtasksperchild=32)
    q = multiprocessing.Manager().Queue()
    for home, dirs, files in os.walk(dir_in_):
        for filename in files:
            abs_path = os.path.join(home, filename)  # 文件绝对路径
            last = p.apply_async(save_img, args=(filename, dir_in_, dir_out_, abs_path, q))
    p.close()
    # p.join()

    pbar = tqdm(total=cnt_all, desc='Convert image')
    for i in range(cnt_all):
        file_name = q.get()
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    # 准备数据集
    # dir_in = r"D:\pic\ob_dataset\test"
    # # dir_in = r"D:\pic\ob_dataset\image_output"
    # dir_out = r"C:\pic\ob_dataset\test--"
    # split_ob_dataset(dir_in, dir_out)
    # # split_ob_dataset_threads(dir_in, dir_out, 6)

    # 生成文件路径列表
    dir_path = r"C:\pic\ob_dataset\sync\baseline_large"
    txt_path = r"filelist-ob-dataset"
    filelist = gen_filelist_txt_ob_dataset(dir_path, txt_path)
    print(len(filelist))

    # 分析数据分布
    # dir_path = r"C:\pic\ob_dataset\sync\baseline_large"
    # hist = analyse_ob_dataset(dir_path)



