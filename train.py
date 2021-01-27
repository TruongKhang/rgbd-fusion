import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = config.init_obj('data_loader', module_data, training=False)
    init_kwags = {
        "data_path": config["data_loader"]["args"]["data_path"],
        "data_list": "lists/dtu/val.txt",
        "mode": "val",
        "num_srcs": config["data_loader"]["args"]["num_srcs"],
        "num_depths": config["data_loader"]["args"]["num_srcs"],
        "interval_scale": config["data_loader"]["args"]["interval_scale"],
        "shuffle": False,
        "seq_size": config["data_loader"]["args"]["seq_size"],
        "batch_size": 1,
        "depth_scale": config["data_loader"]["args"]["depth_scale"]
    }
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(**init_kwags)

    # build models architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    """print('Load pretrained model')
    checkpoint = torch.load('pretrained_model_kitti2.pth')
    new_state_dict = {}
    for key, val in checkpoint['state_dict'].items():
        new_state_dict[key.replace('module.', '')] = val
    model.load_state_dict(new_state_dict, strict=False)
    print('Done')"""

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
