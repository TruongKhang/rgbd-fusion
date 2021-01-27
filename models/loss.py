import math
import torch
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def cfd_loss_decay(outputs, cout, target, epoch_num, *args):
    # val_pixels = torch.ne(target, 0).float() #.cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    err = F.smooth_l1_loss(outputs * val_pixels, gt_depth * val_pixels, reduction='none')
    cert = cout * val_pixels - err * cout * val_pixels
    loss = err - np.exp(-0.1*(epoch_num-1)) * cert
    return torch.sum(loss) / torch.sum(val_pixels)


def cfd_loss_decay_mse(outputs, cout, target, epoch_num, *args):
    # val_pixels = torch.ne(target, 0).float() #.cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    err = F.mse_loss(outputs * val_pixels, gt_depth * val_pixels, reduction='none')
    cert = cout * val_pixels - err * cout * val_pixels
    loss = err - (1 / epoch_num) * cert
    return torch.sum(loss) / torch.sum(val_pixels)


def cfd_loss(outputs, target, cout, *args):
    val_pixels = torch.ne(target, 0).float().cuda()
    err = F.smooth_l1_loss(outputs * val_pixels, target * val_pixels, reduction='none')
    loss = err - cout * val_pixels + err * cout * val_pixels
    return torch.mean(loss)


def smooth_l1_loss(outputs, target, *args):
    # val_pixels = torch.ne(target, 0) #.float().cuda()
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    loss = F.smooth_l1_loss(outputs*val_pixels, gt_depth*val_pixels, reduction='none')
    return torch.sum(loss) / torch.sum(val_pixels)


def rmse_loss(outputs, target, *args):
    val_pixels = (target > 0).float().cuda()
    err = (target * val_pixels - outputs * val_pixels) ** 2
    loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
    cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
    return torch.mean(torch.sqrt(loss / cnt))


def mse_loss(outputs, target, *args):
    gt_depth, val_pixels = target
    val_pixels = val_pixels.float()
    loss = gt_depth * val_pixels - outputs * val_pixels
    return torch.mean(loss ** 2)


def total_loss_l1(outputs1, outputs2, target, cout, epoch_num, ld=0.1):
    loss1 = cfd_loss_decay(outputs1, target, cout, epoch_num)
    loss2 = smooth_l1_loss(outputs2, target)
    return ld * loss1 + (1 - ld) * loss2


def total_loss_mse(outputs1, outputs2, target, cout, epoch_num, ld=0.5):
    loss1 = cfd_loss_decay_mse(outputs1, target, cout, epoch_num)
    loss2 = mse_loss(outputs2, target)
    return ld * loss1 + (1 - ld) * loss2


def masked_l2_gauss(depth, var, targets, *args):
    # (means has shape: (batch_size, 1, h, w))
    # (log_vars has shape: (batch_size, 1, h, w))
    # (targets has shape: (batch_size, 1, h, w))
    gt_depths, valid_mask = targets
    # valid_mask = valid_mask.float()
    init_depth = args[0]
    init_var = args[1]
    scale_factor = args[2]
    epoch = args[3]
    # cnt = gt_depths.size(0) * gt_depths.size(2) * gt_depths.size(3)
    gt = gt_depths[valid_mask]

    init_reg1 = torch.log(init_var + 1e-16)
    init_mean = init_depth[valid_mask]
    init_res = init_var[valid_mask]
    init_reg1 = init_reg1[valid_mask]
    init_loss = torch.mean(init_res * torch.pow(gt - init_mean, 2) - init_reg1)

    reg1 = torch.log(var + 1e-16)
    mean = depth[valid_mask]
    res = var[valid_mask]
    reg1 = reg1[valid_mask]
    final_loss = torch.mean(res * torch.pow(gt - mean, 2) - reg1)

    return final_loss + 0.3*init_loss, None


def masked_prob_loss(depth, var, target):
    gt_depths, valid_mask = target
    # valid_mask = valid_mask.float()
    # cnt = gt_depths.size(0) * gt_depths.size(2) * gt_depths.size(3)
    gt = gt_depths[valid_mask]

    regl = torch.log(var + 1e-16)
    mean = depth[valid_mask]
    res = var[valid_mask]
    regl = regl[valid_mask]
    final_loss = torch.mean(res * torch.pow(gt - mean, 2) - regl)
    return final_loss


class L1_Gradient_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Gradient_loss, self).__init__()
        self.eps = 1e-6
        self.crit = torch.nn.L1Loss(reduction='none')

    def forward(self, X, Y, mask):
        mask_x = mask[:, :, 1:, 1:] - mask[:, :, 0:-1, 1:]
        mask_y = mask[:, :, 1:, 1:] - mask[:, :, 1:, 0:-1]
        valid_mask = (mask_x.eq(0) * mask_y.eq(0)).float() * F.pad(mask, (-1, 0, -1, 0))
        xgin = X[:, :, 1:, 1:] - X[:, :, 0:-1, 1:]
        ygin = X[:, :, 1:, 1:] - X[:, :, 1:, 0:-1]
        xgtarget = Y[:, :, 1:, 1:] - Y[:, :, 0:-1, 1:]
        ygtarget = Y[:, :, 1:, 1:] - Y[:, :, 1:, 0:-1]

        xl = self.crit(xgin, xgtarget)
        yl = self.crit(ygin, ygtarget)
        grad_loss = (xl + yl) * 0.5 * valid_mask
        return torch.mean(grad_loss)


def multi_losses_kitti(depths, cfds, targets, *args):
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    const_depths = args[0]
    coarse_depth = args[1]
    scale_factor = args[2]
    thresh = args[3]

    #gt_depths[gt_depths == 0] = 1e-5
    #error = torch.abs(gt_depths - const_depths)
    #mask = error > thresh * scale_factor
    #error[mask] = gt_depths[mask]
    #gt_cfds = error / gt_depths
    #gt_cfds[gt_cfds > 1] = 1
    #gt_cfds = 1 - gt_cfds
    error = torch.abs(gt_depths - const_depths) / scale_factor
    gt_cfds = torch.exp(-error)

    loss1 = F.mse_loss(depths*valid_mask, gt_depths*valid_mask)
    loss2 = F.mse_loss(cfds*valid_mask, gt_cfds*valid_mask)
    # penalty = torch.mean(cfds*valid_mask ** 2)
    # loss3 = F.l1_loss(coarse_depth*valid_mask, gt_depths*valid_mask)
    # loss4 = F.smooth_l1_loss(itg_depth * valid_mask, gt_depths*valid_mask)
    loss = loss1 + loss2 #+ 0.3*loss3  # + 0.5*loss4
    return loss, gt_cfds


def multi_losses(depths, cfds, targets, *args):
    gt_depths, valid_mask = targets
    valid_mask = valid_mask.float()
    init_depth = args[0]
    init_cfd = args[1]
    scale_factor = args[2]
    epoch = args[3]
    const_depths = depths.detach()
    # gt_depths[gt_depths == 0] = 1e-5
    error = torch.abs(gt_depths - const_depths) / scale_factor
    gt_cfds = torch.exp(-error)

    loss1 = F.mse_loss(depths*valid_mask, gt_depths*valid_mask)
    # grad_loss = L1_Gradient_loss()(depths, gt_depths, valid_mask)
    # loss2 = F.binary_cross_entropy(cfds * valid_mask, gt_cfds * valid_mask) #
    loss2 = F.mse_loss(cfds*valid_mask, gt_cfds*valid_mask)
    # loss3 = F.l1_loss(coarse_depth*valid_mask, gt_depths*valid_mask)
    # loss4 = F.smooth_l1_loss(itg_depth * valid_mask, gt_depths*valid_mask)
    init_error = torch.abs(gt_depths - init_depth.detach()) / scale_factor
    gt_init_cfd = torch.exp(-init_error)
    loss3 = F.mse_loss(init_depth*valid_mask, gt_depths*valid_mask)
    loss4 = F.mse_loss(init_cfd*valid_mask, gt_init_cfd*valid_mask)

    loss = loss1 + loss2 + 0.3*loss3 + loss4
    # cfds = cfds[valid_mask > 0]
    # gt_cfds = gt_cfds[valid_mask > 0]
    # print(cfds.max().item(), cfds.min().item(), cfds.mean().item())
    # print(gt_cfds.max().item(), gt_cfds.min().item(), gt_cfds.mean().item())
    # print(loss1.item(), loss2.item())
    return loss, gt_cfds



