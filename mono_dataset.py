# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import lmdb
import base64
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from boxx import show
import matplotlib.pyplot as plt

from mono_utils.mono_utils import read_depth


class MonoDataset(Dataset):
    def __init__(self, dir_imgs, path_names, mode='train', aug=True, resize_size=None, crop_size=None,
                 degree=0, max_depth=100, dataset_name='kitti', data_idx=[0, 1, 2], mul_times=4,
                 gen_sparse_online=True, sparse_ratio=0.01):
        """
        :param dir_imgs: 数据集所在文件夹
        :param path_names: 包含数据集文件名的txt的路径
        :param mode: 模式，train、val、test
        :param aug: 是否进行数据增强
        :param resize_size: resize的尺寸
        :param crop_size: 裁剪的尺寸
        :param degree: 随机旋转的最大角度
        :param max_depth: 最大深度
        :param dataset_name: 哪种数据集
        :param data_idx: 不同数据的文件名分别在txt的第几列，分别代表img、depth、lidar
        :param mul_times: 图片大小需要是几的倍数才能输入网络
        :param gen_sparse_online: 是否在线对depth进行抽稀得到lidar
        :param sparse_ratio: 抽稀的比例
        """
        super().__init__()
        self.dir_imgs = dir_imgs
        self.aug = aug
        self.mode = mode
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.degree = degree
        self.max_depth = max_depth
        self.dataset_name = dataset_name
        self.gen_sparse_online = gen_sparse_online
        self.mul_times = mul_times  # 输入图片的大小应该为 mul_times 的倍数
        self.idx_img = data_idx[0]  # 代表img数据的文件名在第几列
        self.idx_depth = data_idx[1]  # 代表depth数据的文件名在第几列
        self.idx_lidar = data_idx[2]  # 代表lidar数据的文件名在第几列
        self.data_list = list(pd.read_csv(path_names, header=None)[0])

        # 判断lidar数据是否有效
        self.lidar_exist = True
        self.lidar_persudo = None  # lidar数据无效时用假数据代替
        if not os.path.isfile(os.path.join(self.dir_imgs, self.data_list[0].split()[self.idx_lidar])):
            self.lidar_exist = False
            size = Image.open(os.path.join(self.dir_imgs, self.data_list[0].split()[self.idx_img])).size
            # self.lidar_persudo = Image.new('I', size, 0)
            self.lidar_persudo = np.ones(size[::-1], dtype=np.float32)

        self.img_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.to_tensor = transforms.ToTensor()

        self.to_sparse = iaa.Sequential([iaa.Dropout(1-sparse_ratio),], random_order=True)

    def __getitem__(self, index):
        sample_path = self.data_list[index].split()
        img = Image.open(os.path.join(self.dir_imgs, sample_path[self.idx_img])).convert("RGB")
        lidar = None
        depth = None
        item = []

        if self.mode == 'train':
            depth = read_depth(os.path.join(self.dir_imgs, sample_path[self.idx_depth]), self.dataset_name, self.max_depth)
            if self.lidar_exist:
                lidar = read_depth(os.path.join(self.dir_imgs, sample_path[self.idx_lidar]), self.dataset_name, self.max_depth)
            else:
                if self.gen_sparse_online:
                    lidar = self.to_sparse(image=depth)
                else:
                    lidar = self.lidar_persudo
            # show(depth), show(lidar), show(img)

            # 增强
            rsz_size = img.size[::-1] if self.resize_size is None else self.resize_size  # h*w
            crp_size = img.size[::-1] if self.crop_size is None else self.crop_size  # h*w
            depth_rsz = transforms.Compose([transforms.ToPILImage(), transforms.Resize(rsz_size, 0)])  # 不用插值
            img = transforms.Resize(rsz_size)(img)  # 默认resize为双线性插值
            depth = depth_rsz(depth)
            lidar = depth_rsz(lidar)

            # kitti的上部没有值，先裁剪掉
            if self.dataset_name == 'kitti':
                img = F.crop(img, rsz_size[0] - crp_size[0], 0, crp_size[0], rsz_size[1])
                depth = F.crop(depth, rsz_size[0] - crp_size[0], 0, crp_size[0], rsz_size[1])
                lidar = F.crop(lidar, rsz_size[0] - crp_size[0], 0, crp_size[0], rsz_size[1])

            img = np.asarray(img, dtype=np.float32) / 255.0
            depth = np.asarray(depth)
            lidar = np.asarray(lidar)
            # li = cv2.resize(lidar.astype(np.uint16), rsz_size[::-1], 0)  # opencv的resize会增大稀疏点的比例
            if self.aug:
                img, depth, lidar = self.augment_3(img, depth, lidar, crp_size, self.degree)

            # 标准化
            img = self.img_process(img.copy())
            depth = self.to_tensor(depth.copy())
            lidar = self.to_tensor(lidar.copy())
            item = [img, lidar, depth]
        elif self.mode == 'val':
            depth = read_depth(os.path.join(self.dir_imgs, sample_path[self.idx_depth]), self.dataset_name, self.max_depth)
            if self.lidar_exist:
                lidar = read_depth(os.path.join(self.dir_imgs, sample_path[self.idx_lidar]), self.dataset_name, self.max_depth)
            else:
                if self.gen_sparse_online:
                    lidar = self.to_sparse(image=depth)
                else:
                    lidar = self.lidar_persudo
            img = np.asarray(img, dtype=np.float32) / 255.0

            # pad为网络可输入的大小
            h_ori, w_ori = img.shape[0], img.shape[1]
            h_pad = int(np.ceil(h_ori / self.mul_times) * self.mul_times)
            w_pad = int(np.ceil(w_ori / self.mul_times) * self.mul_times)
            img = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=img)
            lidar = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=lidar)
            depth = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=depth)
            lidar = lidar.astype(np.float32)
            depth = depth.astype(np.float32)

            # 标准化
            img = self.img_process(img.copy())
            depth = self.to_tensor(depth.copy())
            lidar = self.to_tensor(lidar.copy())
            item = [img, lidar, depth]
        elif self.mode == 'test':
            if self.lidar_exist:
                lidar = read_depth(os.path.join(self.dir_imgs, sample_path[self.idx_lidar]), self.dataset_name, self.max_depth)
            else:
                lidar = self.lidar_persudo
            img = np.asarray(img, dtype=np.float32) / 255.0

            # pad为网络可输入的大小
            h_ori, w_ori = img.shape[0], img.shape[1]
            h_pad = int(np.ceil(h_ori / self.mul_times) * self.mul_times)
            w_pad = int(np.ceil(w_ori / self.mul_times) * self.mul_times)
            img = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=img)
            lidar = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=lidar)
            lidar = lidar.astype(np.float32)

            # 标准化
            img = self.img_process(img.copy())
            lidar = self.to_tensor(lidar.copy())
            item = [img, lidar, lidar]
        return item
    
    def __len__(self):
        return len(self.data_list)

    def augment_3(self, image, depth, lidar, crop_size, degree):
        # rsz = iaa.Resize({"height": resize_size[0], "width": resize_size[1]})
        seq = iaa.Sequential([
            iaa.PadToFixedSize(height=crop_size[0], width=crop_size[1]),  # 保证可crop
            iaa.CropToFixedSize(height=crop_size[0], width=crop_size[1]),  # random crop
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            iaa.Rotate((-degree, degree)),
            iaa.GammaContrast((0.9, 1.1)),
            iaa.Multiply((0.9, 1.1)),
        ], random_order=True)
        depth, lidar= np.expand_dims(depth, 2), np.expand_dims(lidar, 2)
        tmp = np.concatenate((depth, lidar), axis=2)
        tmp = (tmp * 1000).astype(np.int32)  # 米单位*1000保留精度
        tmp = SegmentationMapsOnImage(tmp, shape=tmp.shape)
        # image, tmp = rsz(image=image, segmentation_maps=tmp)
        image, tmp = seq(image=image, segmentation_maps=tmp)
        tmp = tmp.arr
        tmp = tmp.astype(np.float32) / 1000  # 再转回米
        depth, lidar = tmp[:,:,0], tmp[:,:,1]
        return image, depth, lidar


class MonoDatasetLMDB(Dataset):
    def __init__(self, path_lmdb, path_names, mode='train', aug=True, resize_size=None, crop_size=None,
                 degree=0, max_depth=100, dataset_name='kitti', data_idx=[0, 2, 1], mul_times=4,
                 gen_sparse_online=True, sparse_ratio=0.01):
        """
        读lmdb数据版
        :param path_lmdb: lmdb数据所在路径
        :param path_names: 包含数据集文件名的txt的路径
        :param mode: 模式，train、val、test
        :param aug: 是否进行数据增强
        :param resize_size: resize的尺寸
        :param crop_size: 裁剪的尺寸
        :param degree: 随机旋转的最大角度
        :param max_depth: 最大深度
        :param dataset_name: 哪种数据集
        :param data_idx: 不同数据的文件名分别在txt的第几列，分别代表img、depth、lidar
        :param mul_times: 图片大小需要是几的倍数才能输入网络
        :param gen_sparse_online: 是否在线对depth进行抽稀得到lidar
        :param sparse_ratio: 抽稀的比例
        """
        super().__init__()
        env = lmdb.open(path_lmdb, max_readers=32, readonly=True)
        self.txn = env.begin()
        self.aug = aug
        self.mode = mode
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.degree = degree
        self.max_depth = max_depth
        self.dataset_name = dataset_name
        self.gen_sparse_online = gen_sparse_online
        self.mul_times = mul_times  # 输入图片的大小应该为 mul_times 的倍数
        self.idx_img = data_idx[0]  # 代表img数据的文件名在第几列
        self.idx_depth = data_idx[1]  # 代表depth数据的文件名在第几列
        self.idx_lidar = data_idx[2]  # 代表lidar数据的文件名在第几列
        self.data_list = list(pd.read_csv(path_names, header=None)[0])

        # 判断lidar数据是否有效
        self.lidar_exist = False
        size = self.load_image(self.data_list[0].split()[self.idx_img]).shape[:2]
        self.lidar_persudo = np.ones(size[::-1], dtype=np.float32)

        self.img_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.to_tensor = transforms.ToTensor()

        self.to_sparse = iaa.Sequential([iaa.Dropout(1 - sparse_ratio), ], random_order=True)

    def __getitem__(self, index):
        sample_path = self.data_list[index].split()
        img = self.load_image(sample_path[self.idx_img])
        lidar = None
        depth = None
        item = []

        if self.mode == 'train':
            depth = self.load_depth(sample_path[self.idx_depth])
            if self.lidar_exist:
                lidar = self.load_depth(sample_path[self.idx_lidar])
            else:
                if self.gen_sparse_online:
                    lidar = self.to_sparse(image=depth)
                else:
                    lidar = self.lidar_persudo
            # show(depth), show(lidar), show(img)

            # 增强
            rsz_size = img.shape[:2] if self.resize_size is None else self.resize_size  # h*w
            crp_size = img.shape[:2] if self.crop_size is None else self.crop_size  # h*w
            img_rsz = transforms.Compose([transforms.ToPILImage(), transforms.Resize(rsz_size)])  # 默认resize为双线性插值
            depth_rsz = transforms.Compose([transforms.ToPILImage(), transforms.Resize(rsz_size, 0)])  # 不用插值
            img = img_rsz(img)
            depth = depth_rsz(depth)
            lidar = depth_rsz(lidar)

            # kitti的上部没有值，先裁剪掉
            if self.dataset_name == 'kitti':
                img = F.crop(img, rsz_size[0] - crp_size[0], 0, crp_size[0], rsz_size[1])
                depth = F.crop(depth, rsz_size[0] - crp_size[0], 0, crp_size[0], rsz_size[1])
                lidar = F.crop(lidar, rsz_size[0] - crp_size[0], 0, crp_size[0], rsz_size[1])

            img = np.asarray(img, dtype=np.float32) / 255.0
            depth = np.asarray(depth)
            lidar = np.asarray(lidar)
            # li = cv2.resize(lidar.astype(np.uint16), rsz_size[::-1], 0)  # opencv的resize会增大稀疏点的比例
            if self.aug:
                img, depth, lidar = self.augment_3(img, depth, lidar, crp_size, self.degree)

            # 标准化
            img = self.img_process(img.copy())
            depth = self.to_tensor(depth.copy())
            lidar = self.to_tensor(lidar.copy())
            item = [img, lidar, depth]
        elif self.mode == 'val':
            depth = self.load_depth(sample_path[self.idx_depth])
            if self.lidar_exist:
                lidar = self.load_depth(sample_path[self.idx_lidar])
            else:
                if self.gen_sparse_online:
                    lidar = self.to_sparse(image=depth)
                else:
                    lidar = self.lidar_persudo
            img = np.asarray(img, dtype=np.float32) / 255.0

            # pad为网络可输入的大小
            h_ori, w_ori = img.shape[0], img.shape[1]
            h_pad = int(np.ceil(h_ori / self.mul_times) * self.mul_times)
            w_pad = int(np.ceil(w_ori / self.mul_times) * self.mul_times)
            img = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=img)
            lidar = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=lidar)
            depth = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=depth)
            lidar = lidar.astype(np.float32)
            depth = depth.astype(np.float32)

            # 标准化
            img = self.img_process(img.copy())
            depth = self.to_tensor(depth.copy())
            lidar = self.to_tensor(lidar.copy())
            item = [img, lidar, depth]
        elif self.mode == 'test':
            if self.lidar_exist:
                lidar = self.load_depth(sample_path[self.idx_lidar])
            else:
                lidar = self.lidar_persudo
            img = np.asarray(img, dtype=np.float32) / 255.0

            # pad为网络可输入的大小
            h_ori, w_ori = img.shape[0], img.shape[1]
            h_pad = int(np.ceil(h_ori / self.mul_times) * self.mul_times)
            w_pad = int(np.ceil(w_ori / self.mul_times) * self.mul_times)
            img = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=img)
            lidar = iaa.CenterPadToFixedSize(height=h_pad, width=w_pad)(image=lidar)
            lidar = lidar.astype(np.float32)

            # 标准化
            img = self.img_process(img.copy())
            lidar = self.to_tensor(lidar.copy())
            item = [img, lidar, lidar]
        return item

    def __len__(self):
        return len(self.data_list)

    def load_image(self, filename):
        Key = '_'.join(((x) for x in (filename[:-4].split('/'))))
        tmp = self.txn.get(Key.encode())
        str_decode = base64.b64decode(tmp)
        # nparr = np.fromstring(str_decode, np.uint8)
        nparr = np.frombuffer(str_decode, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def load_disp(self, filename):
        disparityKey = '_'.join(((x) for x in (filename[:-4].split('/'))))
        fmtKey = '_'.join(((x) for x in (filename[:-4].split('/')))) + '_fmt'
        fmt_tmp = self.txn.get(fmtKey.encode()).decode()
        Size = (int(fmt_tmp.split(',')[0][1:]), int(fmt_tmp.split(',')[1][:-1]))
        buffer = self.txn.get(disparityKey.encode())
        str_decode = base64.b64decode(buffer)
        # nparr = np.fromstring(str_decode, np.uint8)
        nparr = np.frombuffer(str_decode, np.uint8)
        dis_tmp = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        disparityleft = np.zeros((Size[1], Size[0]))
        disparityleft = (dis_tmp[:, :, 0] + 0.01 * dis_tmp[:, :, 1] + 0.0001 * dis_tmp[:, :, 2])
        return np.ascontiguousarray(disparityleft, dtype=np.float32)

    def load_depth(self, filename):
        disp = self.load_disp(filename)
        if self.dataset_name == 'irs':
            disp[disp < 1] = 1
            depth = 48 / disp  # baseline 0.1m, f 480 pixels

        return depth

    def augment_3(self, image, depth, lidar, crop_size, degree):
        # rsz = iaa.Resize({"height": resize_size[0], "width": resize_size[1]})
        seq = iaa.Sequential([
            iaa.PadToFixedSize(height=crop_size[0], width=crop_size[1]),  # 保证可crop
            iaa.CropToFixedSize(height=crop_size[0], width=crop_size[1]),  # random crop
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            iaa.Rotate((-degree, degree)),
            iaa.GammaContrast((0.9, 1.1)),
            iaa.Multiply((0.9, 1.1)),
        ], random_order=True)
        depth, lidar = np.expand_dims(depth, 2), np.expand_dims(lidar, 2)
        tmp = np.concatenate((depth, lidar), axis=2)
        tmp = (tmp * 1000).astype(np.int32)  # 米单位*1000保留精度
        tmp = SegmentationMapsOnImage(tmp, shape=tmp.shape)
        # image, tmp = rsz(image=image, segmentation_maps=tmp)
        image, tmp = seq(image=image, segmentation_maps=tmp)
        tmp = tmp.arr
        tmp = tmp.astype(np.float32) / 1000  # 再转回米
        depth, lidar = tmp[:, :, 0], tmp[:, :, 1]
        return image, depth, lidar


if __name__ == '__main__':
    __spec__ = None

    from torch.utils.data import DataLoader
    from boxx import show

    # dir_imgs_ = "C:\\pic\\ob_dataset\\sync\\baseline_large"
    # path_names_ = "./filelist-ob-dataset-train"
    # dataset = MonoDataset(dir_imgs_, path_names_, resize_size=(360, 640), crop_size=(352, 640),
    #                       mode='train', data_idx=[0,1,2], gen_sparse_online=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # for cnt, (img, lidar, depth) in enumerate(dataloader, 1):
    #     img_show = img.detach().numpy()[0, 0, :, :]
    #     # depth_gt_show = depth_gt.detach().numpy()[0, 0, :, :]
    #     # show(img, depth, lidar)
    #     # show(img, lidar)
    #     a = 1

    # dir_imgs_ = r"D:\pic\KITTI\sync"
    # path_names_ = r"./filelist-kitti-dataset-train"
    # dataset = MonoDataset(dir_imgs_, path_names_, crop_size=(256, 1216), mode='train', data_idx=[0, 1, 2])
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # for cnt, (img, lidar, depth_gt) in enumerate(dataloader, 1):
    #     a = 1

    # # dir_imgs_ = r"D:\pic\IRS"
    # # path_names_ = r"./filelist-irs-s-dataset-train"
    # dir_imgs_ = r"E:\data\home\xubin\dataset\dataset\IRSDataset"
    # path_names_ = r"./filelist-irs-dataset-train"
    # dataset = MonoDataset(dir_imgs_, path_names_, dataset_name='irs', crop_size=(512, 960), mode='train', data_idx=[0, 1, 2])
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # for cnt, (img, lidar, depth_gt) in enumerate(dataloader, 1):
    #     a = 1

    dir_imgs_ = r"E:\data\home\yangxiaoli\stereo_data\IRSDataset_lmdb"
    path_names_ = r"./filelist-lmdb-subset-irs-train"
    dataset = MonoDatasetLMDB(dir_imgs_, path_names_, dataset_name='irs', crop_size=(512, 960), mode='train')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for cnt, (img, lidar, depth_gt) in enumerate(dataloader, 1):
        a = 1
