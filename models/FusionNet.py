"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from .ERFNet import Net


class uncertainty_net(nn.Module):
    def __init__(self, in_channels, out_channels=1, thres=15):
        super(uncertainty_net, self).__init__()
        out_chan = 2

        combine = 'concat'
        self.combine = combine
        self.in_channels = in_channels

        out_channels = 3
        self.depthnet = Net(in_channels=in_channels, out_channels=out_channels)

        local_channels_in = 2 if self.combine == 'concat' else 1
        self.convbnrelu = nn.Sequential(convbn(local_channels_in, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))
        self.hourglass1 = hourglass_1(32)
        self.hourglass2 = hourglass_2(32)
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, out_chan, kernel_size=3, padding=1, stride=1, bias=True))
        self.activation = nn.ReLU(inplace=True)
        self.thres = thres
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, epoch=50):
        if self.in_channels > 1:
            rgb_in = input[:, 1:, :, :]
            lidar_in = input[:, 0:1, :, :]
        else:
            lidar_in = input

        # 1. GLOBAL NET
        embedding0, embedding1, embedding2 = self.depthnet(input)

        global_features = embedding0[:, 0:1, :, :]
        precise_depth = embedding0[:, 1:2, :, :]
        conf = embedding0[:, 2:, :, :]

        # 2. Fuse 
        if self.combine == 'concat':
            input = torch.cat((lidar_in, global_features), 1)
        elif self.combine == 'add':
            input = lidar_in + global_features
        elif self.combine == 'mul':
            input = lidar_in * global_features
        elif self.combine == 'sigmoid':
            input = lidar_in * nn.Sigmoid()(global_features)
        else:
            input = lidar_in

        # 3. LOCAL NET
        out = self.convbnrelu(input)
        out1, embedding3, embedding4 = self.hourglass1(out, embedding1, embedding2)
        out1 = out1 + out
        out2 = self.hourglass2(out1, embedding3, embedding4)
        out2 = out2 + out
        out = self.fuse(out2)
        lidar_out = out

        # 4. Late Fusion
        lidar_to_depth, lidar_to_conf = torch.chunk(out, 2, dim=1)
        lidar_to_conf, conf = torch.chunk(self.softmax(torch.cat((lidar_to_conf, conf), 1)), 2, dim=1)
        out = conf * precise_depth + lidar_to_conf * lidar_to_depth

        output = [out, lidar_out, precise_depth, global_features]
        return output


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
                         # nn.BatchNorm2d(out_planes))


class hourglass_1(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_1, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, em1), 1)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = F.relu(x_prime, inplace=True)
        x_prime = torch.cat((x_prime, em2), 1)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out, x, x_prime


class hourglass_2(nn.Module):
    def __init__(self, channels_in):
        super(hourglass_2, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + em1
        x = F.relu(x, inplace=True)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out 


class MAE_loss(nn.Module):
    def __init__(self):
        super(MAE_loss, self).__init__()

    def forward(self, prediction, gt, input, epoch=0):
        prediction = prediction[:, 0:1]
        abs_err = torch.abs(prediction - gt)
        mask = (gt > 0).detach()
        mae_loss = torch.mean(abs_err[mask])
        return mae_loss


class MAE_log_loss(nn.Module):
    def __init__(self):
        super(MAE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        prediction = torch.clamp(prediction, min=0)
        abs_err = torch.abs(torch.log(prediction+1e-6) - torch.log(gt+1e-6))
        mask = (gt > 0).detach()
        mae_log_loss = torch.mean(abs_err[mask])
        return mae_log_loss


class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        err = prediction[:,0:1] - gt
        mask = (gt > 0).detach()
        mse_loss = torch.mean((err[mask])**2)
        return mse_loss


class MSE_loss_uncertainty(nn.Module):
    def __init__(self):
        super(MSE_loss_uncertainty, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        mask = (gt > 0).detach()
        depth = prediction[:, 0:1, :, :]
        conf = torch.abs(prediction[:, 1:, :, :])
        err = depth - gt
        conf_loss = torch.mean(0.5*(err[mask]**2)*torch.exp(-conf[mask]) + 0.5*conf[mask])
        return conf_loss


class MSE_log_loss(nn.Module):
    def __init__(self):
        super(MSE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        prediction = torch.clamp(prediction, min=0)
        err = torch.log(prediction+1e-6) - torch.log(gt+1e-6)
        mask = (gt > 0).detach()
        mae_log_loss = torch.mean(err[mask]**2)
        return mae_log_loss


class Huber_loss(nn.Module):
    def __init__(self, delta=10):
        super(Huber_loss, self).__init__()
        self.delta = delta

    def forward(self, outputs, gt, input, epoch=0):
        outputs = outputs[:, 0:1, :, :]
        err = torch.abs(outputs - gt)
        mask = (gt > 0).detach()
        err = err[mask]
        squared_err = 0.5*err**2
        linear_err = err - 0.5*self.delta
        return torch.mean(torch.where(err < self.delta, squared_err, linear_err))



class Berhu_loss(nn.Module):
    def __init__(self, delta=0.05):
        super(Berhu_loss, self).__init__()
        self.delta = delta

    def forward(self, prediction, gt, epoch=0):
        prediction = prediction[:, 0:1]
        err = torch.abs(prediction - gt)
        mask = (gt > 0).detach()
        err = torch.abs(err[mask])
        c = self.delta*err.max().item()
        squared_err = (err**2+c**2)/(2*c)
        linear_err = err
        return torch.mean(torch.where(err > c, squared_err, linear_err))


class Huber_delta1_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, gt, input):
        mask = (gt > 0).detach().float()
        loss = F.smooth_l1_loss(prediction*mask, gt*mask, reduction='none')
        return torch.mean(loss)


class Disparity_Loss(nn.Module):
    def __init__(self, order=2):
        super(Disparity_Loss, self).__init__()
        self.order = order

    def forward(self, prediction, gt):
        mask = (gt > 0).detach()
        gt = gt[mask]
        gt = 1./gt
        prediction = prediction[mask]
        err = torch.abs(prediction - gt)
        err = torch.mean(err**self.order)
        return err


if __name__ == '__main__':
    batch_size = 4
    in_channels = 4
    H, W = 256, 1216
    model = uncertainty_net(34, in_channels).cuda()
    print(model)
    print("Number of parameters in model is {:.3f}M".format(sum(tensor.numel() for tensor in model.parameters())/1e6))
    input = torch.rand((batch_size, in_channels, H, W)).cuda().float()
    out = model(input)
    print(out.shape)
