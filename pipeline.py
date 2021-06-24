import torch
import torch.nn as nn

import argparse
import os
import pprint
import shutil

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from core.config import get_model_name
from dataset import coco

import timm
from swinTposeModel import KeypointDetector as MyNet
from swinTposeModel import swintBackbone as model_backbone
from swinTposeModel import DeconvHead as model_head

# KeypointDetector(swintBackbone(pretrained=False),DeepLabV3Head())

# class MyNet(nn.Module):
#     def __init__(self, args):
#         super(MyNet, self).__init__()
#         self.net = nn.Conv2d(3, 17, kernel_size=5, stride=4,
#                      padding=1, bias=False)
#         # self.swin_backbone = swinT224() if args.img_size == 224 else swinT384
#         self.swin_backbone = timm.create_model('swin_base_patch4_window{}_{}_in22k'.format(
#             args.img_size//32, args.img_size), pretrained=True)
#
#     def forward(self, x):
#         out = self.net(x)
#         # out = self.swin_backbone(x)
#         return out


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cuda', default=2, type=int)
    parser.add_argument('--num-workers', default=4, type=int)

    parser.add_argument('--batch-size', default=8, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=100, type=int) # unused
    parser.add_argument('--weight-decay', default=1e-4, type=float)

    parser.add_argument('--img-size', default=224, type=int) # [224, 384] is supported
    parser.add_argument('--htmp-ratio', default=4, type=int) # output size is 1/htmp-ratio of the original

    parser.add_argument('--save-interval', default=10, type=int)
    parser.add_argument('--resume', default='None', type=str) # unused

    parser.add_argument('--save-no', default=8, type=int)

    # you need to output the det result to json at inference stage.
    parser.add_argument('--bbox_json', default='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json', type=str)

    parser.add_argument('--log-name', default='formal', type=str)
    parser.add_argument('--print-freq', default=4000, type=int)
    parser.add_argument('--vis', default=False, action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    model_name = 'swin_small_patch4_window{}_{}'.format(args.img_size//32, args.img_size)
    BaseNet = MyNet(model_backbone(name=model_name, pretrained=True), model_head(down_scale=args.htmp_ratio, in_channels=768))
    BaseNet = BaseNet.cuda(args.cuda)

    config = {'OUTPUT_DIR': 'output', 'LOG_DIR': 'log', 'DATA_DIR': '', 'GPU': args.cuda, 'WORKERS': args.num_workers, 'PRINT_FREQ': args.print_freq,
              'CUDNN': {'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True},
              'MODEL': {'NAME': 'MyNet', 'INIT_WEIGHTS': True, 'PRETRAINED': 'None', 'NUM_JOINTS': 17, 'IMAGE_SIZE': np.array((args.img_size, args.img_size)),
                        'EXTRA': {'TARGET_TYPE': 'gaussian', 'HEATMAP_SIZE': np.array((args.img_size // args.htmp_ratio, args.img_size // args.htmp_ratio)), 'SIGMA': 2, 'FINAL_CONV_KERNEL': 1, 'DECONV_WITH_BIAS': False, 'NUM_DECONV_LAYERS': 3, 'NUM_LAYERS': args.save_no}, 'STYLE': 'pytorch'},
              'LOSS': {'USE_TARGET_WEIGHT': True}, 'DATASET': {'ROOT': 'data/coco/', 'DATASET': 'coco', 'TRAIN_SET': 'train2017', 'TEST_SET': 'val2017', 'DATA_FORMAT': 'jpg', 'HYBRID_JOINTS_TYPE': '', 'SELECT_DATA': False, 'FLIP': True, 'SCALE_FACTOR': 0.3, 'ROT_FACTOR': 40},
              'TRAIN': {'LR_FACTOR': 0.1, 'LR_STEP': [40, 80], 'LR': 0.001, 'OPTIMIZER': 'adam', 'MOMENTUM': 0.9, 'WD': 0.0001, 'NESTEROV': False, 'GAMMA1': 0.99, 'GAMMA2': 0.0, 'BEGIN_EPOCH': 0, 'END_EPOCH': 140, 'RESUME': False, 'CHECKPOINT': '', 'BATCH_SIZE': 32, 'SHUFFLE': True},
              'TEST': {'BATCH_SIZE': args.batch_size, 'FLIP_TEST': False, 'POST_PROCESS': True, 'SHIFT_HEATMAP': True, 'USE_GT_BBOX': True, 'OKS_THRE': 0.9, 'IN_VIS_THRE': 0.2, 'COCO_BBOX_FILE': args.bbox_json, 'BBOX_THRE': 1.0, 'MODEL_FILE': '', 'IMAGE_THRE': 0.0, 'NMS_THRE': 1.0},
              'DEBUG': {'DEBUG': args.vis, 'SAVE_BATCH_IMAGES_GT': args.vis, 'SAVE_BATCH_IMAGES_PRED': args.vis, 'SAVE_HEATMAPS_GT': args.vis, 'SAVE_HEATMAPS_PRED': args.vis}}

    config = edict(config)

    criterion = JointsMSELoss(use_target_weight=True).cuda(args.cuda)

    optimizer = torch.optim.Adam(BaseNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.log_name, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    train_dataset = coco(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = coco(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, args.num_epoch):
        lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, BaseNet, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, BaseNet,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        if epoch % args.save_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': BaseNet.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, filename='checkpoint{}.pth.tar'.format(str(epoch).zfill(3)))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(BaseNet.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__=="__main__":
    main()