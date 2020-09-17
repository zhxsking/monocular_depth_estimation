# -*- coding: utf-8 -*-

import sys
import os
import shutil
import time
import copy
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from boxx import g, show

from models.bts import *
from models.bts_v1 import *
from models.FusionNet import *
from mono_dataset import *
from mono_utils.mono_utils import *


def evalNet(model, dataloader_val, device, mea_nums):
    """用验证集评判网络性能"""
    model.eval()
    metrics_val = Record(mea_nums)
    with torch.no_grad():
        for cnt, (img, lidar_in, depth_gt) in enumerate(tqdm(dataloader_val), 1):
            img = img.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)

            output = model(img)
            depth_est, _, _, _, _ = output[0], output[1], output[2], output[3], output[4]
            depth_est = torch.clamp(depth_est, 0, args.max_depth)

            valid_mask = get_valid_mask(depth_gt, args)
            metrics_tmp, min_is_best_mask, metrics_names = compute_errors_9(depth_gt, depth_est, valid_mask)
            metrics_val.update(metrics_tmp, img.shape[0])

    return metrics_val.avg, min_is_best_mask, metrics_names


def evalNet_lidar(model, dataloader_val, device, mea_nums):
    """用验证集评判网络性能"""
    model.eval()
    metrics_val = Record(mea_nums)
    with torch.no_grad():
        for cnt, (img, lidar_in, depth_gt) in enumerate(tqdm(dataloader_val), 1):
            img = img.to(device, non_blocking=True)
            lidar_in = lidar_in.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)

            input = torch.cat((lidar_in, img), 1)
            output = model(input)
            depth_est, _, _, _ = output[0], output[1], output[2], output[3]
            depth_est = torch.clamp(depth_est, 0, args.max_depth)

            # output = model([img, lidar_in])
            # depth_est, _, _, _, _ = output[0], output[1], output[2], output[3], output[4]
            # depth_est = torch.clamp(depth_est, 0, args.max_depth)

            valid_mask = get_valid_mask(depth_gt, args)
            metrics_tmp, min_is_best_mask, metrics_names = compute_errors_3(depth_gt, depth_est, valid_mask)
            metrics_val.update(metrics_tmp, img.shape[0])

    return metrics_val.avg, min_is_best_mask, metrics_names


if __name__ == '__main__':
    __spec__ = None

    parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--data_path', type=str, help='path to the data', required=True)
    parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu or ob', default='nyu')
    parser.add_argument('--filelist_txt', type=str, help='path to the filenames text file', required=True)
    parser.add_argument('--filelist_txt_val', type=str, help='path to the filenames text file', required=True)
    parser.add_argument('--batchsize', type=int, help='batch size', default=4)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_end', type=float, help='initial learning rate', default=1e-5)
    parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-4)
    parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--weight_init', type=str, default='kaiming',
                        help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--sparse_ratio', type=float, help='sparse_ratio', default=0.01)
    parser.add_argument('--lr_policy', type=str, default=None, help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=0)
    parser.add_argument('--do_eigen_crop', help='if do eigen crop to test', action='store_true')

    parser.add_argument('--resize_w', type=int, help='resize width', default=None)
    parser.add_argument('--resize_h', type=int, help='resize height', default=None)
    parser.add_argument('--crop_w', type=int, help='crop width', default=None)
    parser.add_argument('--crop_h', type=int, help='crop height', default=None)

    parser.add_argument('--gpu', type=int, help='GPU id to use.', default=None)
    parser.add_argument('--variance_focus', type=float, help='lambda in paper: [0, 1]', default=0.85)
    parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
    parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=100)
    parser.add_argument('--val_freq', type=int, help='validation frequency in global steps', default=500)
    parser.add_argument('--not_save_checkpoint', help='if set, will not save checkpoint', action='store_true')
    parser.add_argument('--log_dir', type=str, help='directory to save checkpoints and summaries', default='')
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
    rsz_size = None if args.resize_h is None or args.resize_w is None else (args.resize_h, args.resize_w)
    crp_size = None if args.crop_h is None or args.crop_w is None else (args.crop_h, args.crop_w)
    dataset_train = MonoDataset(args.data_path, args.filelist_txt, resize_size=rsz_size, crop_size=crp_size,
                                degree=args.degree, max_depth=args.max_depth, dataset_name=args.dataset, mul_times=32,
                                sparse_ratio=args.sparse_ratio)
    dataset_val = MonoDataset(args.data_path, args.filelist_txt_val, resize_size=rsz_size, crop_size=crp_size,
                              degree=args.degree, max_depth=args.max_depth, dataset_name=args.dataset, mul_times=32,
                              sparse_ratio=args.sparse_ratio, mode='val')

    # dataset_train = MonoDatasetLMDB(args.data_path, args.filelist_txt, resize_size=rsz_size, crop_size=crp_size,
    #                                 degree=args.degree, max_depth=args.max_depth, dataset_name=args.dataset,
    #                                 mul_times=32, sparse_ratio=args.sparse_ratio)
    # dataset_val = MonoDatasetLMDB(args.data_path, args.filelist_txt_val, resize_size=rsz_size, crop_size=crp_size,
    #                               degree=args.degree, max_depth=args.max_depth, dataset_name=args.dataset,
    #                               mul_times=32, sparse_ratio=args.sparse_ratio, mode='val')

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batchsize, shuffle=True,
                                  num_workers=args.num_threads, pin_memory=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                num_workers=args.num_threads, pin_memory=True)

    # 定义网络等
    device_str = "cpu" if args.gpu is None else "cuda:{}".format(args.gpu)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # model = BtsModel(args.max_depth, args.encoder).to(device)
    # set_misc(model)
    # loss_func = silog_loss(variance_focus=args.variance_focus).to(device)
    # optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'weight_decay': args.weight_decay},
    #                                {'params': model.decoder.parameters(), 'weight_decay': 0}],
    #                               lr=args.lr, eps=1e-6)
    # mea_nums = 9

    model = uncertainty_net(in_channels=4).to(device)
    define_init_weights(model, args.weight_init)
    optimizer = define_optim(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    criterion_local = MSE_loss()
    criterion_lidar = MSE_loss()
    criterion_rgb = MSE_loss()
    criterion_guide = MSE_loss()
    mea_nums = 3

    # model = BtsModel_v1(args.max_depth).to(device)
    # optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'weight_decay': args.weight_decay},
    #                                {'params': model.decoder.parameters(), 'weight_decay': 0}],
    #                               lr=args.lr, eps=1e-6)
    # loss_func = MSE_loss()
    # mea_nums = 3

    # 初始化
    os.makedirs(os.path.join(args.log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'summary'), exist_ok=True)
    shutil.copy(sys.argv[1], args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, 'models-py')):
        shutil.copytree('models', os.path.join(args.log_dir, 'models-py'))
    shutil.copy('mono_train.py', args.log_dir)
    shutil.copy('mono_dataset.py', args.log_dir)

    global_step = 1
    loss_mea = Record()
    metrics_mea = Record(mea_nums)
    metrics_train = Record(mea_nums)
    best_mea = np.zeros(mea_nums, dtype=np.float32)
    measures_names = []
    nan_cnt = 0
    cur_lr = args.lr
    last_model = {'lr': cur_lr, 'model': copy.deepcopy(model.state_dict())}  # 用于碰到nan情况时进行恢复

    log = Logger()
    log.open(os.path.join(args.log_dir, 'log.txt'))
    if args.msg != '':
        log.write(args.msg, True)
    log.write(device_str, True)
    log.write('Loaded Model of {}'.format(model.__class__), True)
    log.write('Loading Data of [{}], train set: {}, val set: {}'.
              format(args.data_path, len(dataset_train), len(dataset_val)), True)

    sfx = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    writer = SummaryWriter(os.path.join(args.log_dir, 'summary', sfx), flush_secs=30)

    # 读取checkpoint
    just_load_net = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_mea = checkpoint['best_measures']
            just_load_net = True
            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
    steps_per_epoch = len(dataloader_train)
    steps_total = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    scheduler = define_scheduler(optimizer, args.lr_policy, gamma=0.5, step_size=7,
                                 end_lr=args.lr_end, step_all=steps_total)

    # 训练
    print('Start Training...')
    start_time = time.time()  # 记录时间
    model.train()
    while epoch < args.num_epochs:
        for step, (img, lidar_in, depth_gt) in enumerate(dataloader_train, 1):
            img = img.to(device, non_blocking=True)
            lidar_in = lidar_in.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)
            # show(img), show(lidar_in), show(depth_gt), show(depth_est)

            # output = model(img)
            # depth_est, lpg8x8, lpg4x4, lpg2x2, reduc1x1 = output[0], output[1], output[2], output[3], output[4]
            # depth_est = torch.clamp(depth_est, 0, args.max_depth)
            # mask = depth_gt > 0
            # loss = loss_func(depth_est, depth_gt, mask.to(torch.bool))
            # for param_group in optimizer.param_groups:
            #     current_lr = (args.lr - args.lr_end) * (1 - global_step / steps_total) ** 0.9 + args.lr_end
            #     param_group['lr'] = current_lr

            input = torch.cat((lidar_in, img), 1).to(device, non_blocking=True)
            output = model(input)
            depth_est, lidar_out, precise, guide = output[0], output[1], output[2], output[3]
            depth_est = torch.clamp(depth_est, 0, args.max_depth)
            loss = criterion_local(depth_est, depth_gt)
            loss_lidar = criterion_lidar(lidar_out, depth_gt)
            loss_rgb = criterion_rgb(precise, depth_gt)
            loss_guide = criterion_guide(guide, depth_gt)
            loss = 1 * loss + 0.1 * loss_lidar + 0.1 * loss_rgb + 0.1 * loss_guide

            # output = model([img, lidar_in])
            # depth_est, lpg8x8, lpg4x4, lpg2x2, reduc1x1 = output[0], output[1], output[2], output[3], output[4]
            # depth_est = torch.clamp(depth_est, 0, args.max_depth)
            # loss = loss_func(depth_est, depth_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid_mask = get_valid_mask(depth_gt, args)
            metrics_tmp_train, _, metrics_names_train = compute_errors_3(depth_gt, depth_est, valid_mask)
            metrics_train.update(metrics_tmp_train, img.shape[0])
            loss_mea.update(loss.item(), img.shape[0])

            # 遇到nan，则恢复之前正常的网络，同时重置lr
            if np.isnan(loss.cpu().item()):
                # nan_cnt += 1
                # model.load_state_dict(last_model['model'])
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = args.lr
                print('Nan appear!')
                nan_net_state = {'global_step': global_step, 'model': model.state_dict()}
                nan_net_path = os.path.join(args.log_dir, 'models-nan')
                torch.save(nan_net_state, nan_net_path)
                sys.exit(0)

            cur_lr = optimizer.param_groups[0]['lr']
            if args.lr_policy is not None and args.lr_policy != 'plateau':
                scheduler.step()
            print('[epoch][step]: [{}/{}][{}/{}/{}], lr: {:.4e}, loss: {:.6f}, rmse: {:.6f}, ard: {:.6f}'.
                  format(epoch+1, args.num_epochs, step, steps_per_epoch, global_step, cur_lr, loss.item(),
                         metrics_tmp_train[0], metrics_tmp_train[2]))

            # 定时输出信息
            if global_step % args.log_freq == 0:
                time_elapsed = time.time() - start_time
                training_time_left = (steps_total / global_step - 1.0) * time_elapsed
                msg = '[epoch][step]: [{}/{}][{}/{}/{}], lr: {:.4e}, loss: {:.6f}, rmse: {:.6f}, ard: {:.6f}, ' \
                      'time elapsed: {:.0f}h {:.0f}m {:.2f}s, need: {:.0f}h {:.0f}m {:.2f}s'.\
                    format(epoch+1, args.num_epochs, step, steps_per_epoch, global_step, cur_lr,
                           loss_mea.avg[0], metrics_train.avg[0], metrics_train.avg[2],
                           time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60,
                           training_time_left // 3600, training_time_left // 60 % 60, training_time_left % 60)
                log.write(msg, True)

                writer.add_scalar('loss', loss_mea.avg[0], global_step)
                writer.add_scalar('lr', cur_lr, global_step)
                grid_depth_est = torchvision.utils.make_grid(linear_p_tensor(depth_est))
                grid_depth_gt = torchvision.utils.make_grid(linear_p_tensor(depth_gt))
                grid_lidar = torchvision.utils.make_grid(lidar_in, normalize=True, scale_each=True)
                grid_img = torchvision.utils.make_grid(img, normalize=True, scale_each=True)
                writer.add_image('depth-pred', grid_depth_est, global_step)
                writer.add_image('depth-gt', grid_depth_gt, global_step)
                writer.add_image('img', grid_img, global_step)
                writer.add_image('lidar', grid_lidar, global_step)
                writer.add_histogram('depth_gt_hist', depth_gt.clone().detach().cpu().numpy(), global_step)
                writer.add_histogram('depth_pred_hist', depth_est.clone().detach().cpu().numpy(), global_step)
                # for name, param in model.named_parameters():
                #     writer.add_histogram(name, param.clone().detach().cpu().numpy(), global_step)
                writer.flush()

                loss_mea.save_avg_list()
                loss_mea.reset()

            # 验证并保存最佳网络
            if global_step % args.val_freq == 0 and not just_load_net or global_step == steps_total:
                eval_measures, min_is_best_mask, measures_names = evalNet_lidar(model, dataloader_val, device, mea_nums)
                # eval_measures, min_is_best_mask, measures_names = evalNet(model, dataloader_val, device, mea_nums)
                model.train()

                for i in range(len(eval_measures)):
                    writer.add_scalar(measures_names[i], eval_measures[i], global_step)
                    writer.flush()

                if args.lr_policy == 'plateau':
                    scheduler.step(eval_measures[0])
                msg = get_measures_msg(eval_measures, measures_names)
                log.write(device_str + ', ' + msg, True)
                for i in range(len(eval_measures)):
                    # 第一次不比较
                    if 0 == best_mea.sum():
                        best_mea = eval_measures
                        break

                    is_best = False
                    if min_is_best_mask[i] and eval_measures[i] < best_mea[i]:
                        best_mea[i] = eval_measures[i]
                        is_best = True
                    if not(min_is_best_mask[i]) and eval_measures[i] > best_mea[i]:
                        best_mea[i] = eval_measures[i]
                        is_best = True
                    if is_best and not args.not_save_checkpoint:
                        last_model = {'lr': cur_lr, 'model': copy.deepcopy(model.state_dict())}
                        for mdl_name in os.listdir(os.path.join(args.log_dir, 'models')):
                            if measures_names[i] in mdl_name:
                                os.remove(os.path.join(args.log_dir, 'models', mdl_name))
                        best_net_state = {
                            'global_step': global_step,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_measures': best_mea,
                        }
                        net_name = 'model-{}-best-{}-{:.5f}'.format(global_step, measures_names[i], eval_measures[i])
                        best_net_path = os.path.join(args.log_dir, 'models', net_name)
                        print('New best {}. Saving model: {}'.format(measures_names[i], net_name))
                        torch.save(best_net_state, best_net_path)
            just_load_net = False
            global_step += 1
        epoch += 1
    torch.cuda.empty_cache()

    # 保存最终模型以及参数
    time_elapsed = time.time() - start_time  # 用时
    final_net_state = {
        'time': time_elapsed,
        'global_step': global_step - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_measures': best_mea,
    }
    final_net_path = os.path.join(args.log_dir, 'models', 'final-net-{}'.format(global_step-1))
    torch.save(final_net_state, final_net_path)

    # 显示训练信息
    print('-' * 50)
    msg = ('Training complete in {:.0f}h {:.0f}m {:.2f}s'.
           format(time_elapsed // 3600, time_elapsed // 60 % 60, time_elapsed % 60))
    log.write(msg, True)

    best_mea_msg = 'Best '
    for i in range(len(best_mea)):
        best_mea_msg += '{}: {}, '.format(measures_names[i], best_mea[i])
    log.write(best_mea_msg, True)
    log.close()
