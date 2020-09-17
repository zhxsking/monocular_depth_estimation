# -*- coding: utf-8 -*-

import os
import shutil
from tqdm import tqdm


def proc_sync_rgb(dir_rgb_):
    """处理kitti原始数据集的RGB部分，删除前5和后5张图像，保证跟官方处理后的depth数据一致"""
    cnt_all = 0
    for home, dirs, files in os.walk(dir_rgb_):
        cnt_all += len(files)

    # 记录需要删除的文件夹和图片
    dir_need_rm = []
    file_need_rm = []
    pbar = tqdm(total=cnt_all, desc='Get lists')
    for home, dirs, files in os.walk(dir_rgb_):
        if home.split("\\")[-1] == 'oxts' or home.split("\\")[-1] == 'velodyne_points':
            dir_need_rm.append(home)

        for filename in files:
            pbar.update(1)
            abs_path = os.path.join(home, filename)  # 文件绝对路径

            # 转换成对应的GT路径，验证对应的GT是否存在
            gt_path = (abs_path.
                       replace('data_rgb', 'data_depth').
                       replace('image', 'proj_depth\\groundtruth\\image').
                       replace('image_00', 'image_02').
                       replace('image_01', 'image_02').
                       replace('data\\00', '00'))
            if not os.path.isfile(gt_path):
                file_need_rm.append(abs_path)
    pbar.close()

    # 删除文件夹和图片
    for dir_ in tqdm(dir_need_rm):
        shutil.rmtree(dir_)
    for file in tqdm(file_need_rm):
        os.remove(file)

    return dir_need_rm, file_need_rm


if __name__ == "__main__":
    dir_rgb = r'D:\pic\KITTI\sync\data_rgb'
    proc_sync_rgb(dir_rgb)



