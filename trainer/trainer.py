import numpy as np
import os
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, util
import MYTH


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.data_loader.set_device(self.device)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['logging_every'] # int(np.sqrt(data_loader.batch_size))

        name_metrics = list()
        for m in self.metric_ftns:
            name_metrics.append(m.__name__)
        self.train_metrics = MetricTracker('loss', *name_metrics, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *name_metrics, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.data_loader.shuffle_and_crop()
        for batch_idx, sample in enumerate(self.data_loader):
            sample_cuda = util.tocuda(sample)

            depths, confs = sample_cuda["input_depths"], sample_cuda["input_confs"]
            proj_matrices = sample_cuda["proj_matrices"]["stage3"]
            intrinsics, extrinsics = proj_matrices[:, :, 1, :, :], proj_matrices[:, :, 0, :, :]
            camera_params = torch.matmul(intrinsics[..., :3, :3], extrinsics[..., :3, :4])
            warped_depths, warped_confs, _ = MYTH.DepthColorAngleReprojectionNeighbours.apply(depths, confs, camera_params, 1.0)
            ref_depth, src_depths = warped_depths[:, 0, ...], warped_depths[:, 1:, ...]
            ref_conf, src_confs = warped_confs[:, 0, ...], warped_confs[:, 1:, ...]

            self.optimizer.zero_grad()
            pred_depth, pred_conf = self.model(src_depths, src_confs)

            gt_depth = sample_cuda["depth"]["stage3"].unsqueeze(1)
            mask = sample_cuda["depth"]["stage3"].unsqueeze(1)
            target = (gt_depth, mask > 0.5)
            loss = self.criterion(pred_depth, pred_conf, target)
            loss.backward()

            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), n=target[0].size(0))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(pred_depth, target).item(), n=target[0].size(0))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}, #processed_frames: {} Loss: {:.6f}, RMSE: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.train_metrics.avg('loss'),
                    self.train_metrics.avg('rmse')))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch, save_folder=None)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, save_folder=None):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation at epoch %d, size of validation set: %d, batch_size: %d" % (epoch, len(self.valid_data_loader),
                                                                                     self.valid_data_loader.batch_size))
        if save_folder is not None:
            path_depth = os.path.join(save_folder, 'depth_maps')
            if not os.path.exists(path_depth):
                os.makedirs(path_depth)
            path_cfd = os.path.join(save_folder, 'confidence')
            if not os.path.exists(path_cfd):
                os.makedirs(path_cfd)

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                sample_cuda = util.tocuda(sample)

                depths, confs = sample_cuda["input_depths"], sample_cuda["input_confs"]
                proj_matrices = sample_cuda["proj_matrices"]["stage3"]
                intrinsics, extrinsics = proj_matrices[:, :, 1, :, :], proj_matrices[:, :, 0, :, :]
                camera_params = torch.matmul(intrinsics[..., :3, :3], extrinsics[..., :3, :4])
                warped_depths, warped_confs, _ = MYTH.DepthColorAngleReprojectionNeighbours.apply(depths, confs,
                                                                                                  camera_params, 1.0)
                ref_depth, src_depths = warped_depths[:, 0, ...], warped_depths[:, 1:, ...]
                ref_conf, src_confs = warped_confs[:, 0, ...], warped_confs[:, 1:, ...]

                pred_depth, pred_conf = self.model(src_depths, src_confs)

                gt_depth = sample_cuda["depth"]["stage3"].unsqueeze(1)
                mask = sample_cuda["depth"]["stage3"].unsqueeze(1)
                target = (gt_depth, mask > 0.5)
                loss = self.criterion(pred_depth, pred_conf, target)

                # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.valid_metrics.update('loss', loss.item(), n=target[0].size(0))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(pred_depth, target).item(), n=target[0].size(0))

                if save_folder is not None:
                    util.save_image(path_depth, '%d.png' % batch_idx, pred_depth.squeeze(0).squeeze(0).cpu().numpy())
                    util.save_image(path_cfd, '%d.png' % batch_idx, pred_conf.squeeze(0).squeeze(0).cpu().numpy())

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.get_num_samples()
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
