import os
import gc
import time
import datetime
import operator
from functools import reduce

import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F

import utils
import transforms as T
from lib import lgformer
from data.dataset_refer_bert import ReferDataset


def get_dataset(image_set, transform, args, eval_mode=False):
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=eval_mode
                      )
    return ds


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def criterion(preds, targets):
    """
    Args:
        preds: (B, 1, 480, 480)
        targets: (B, 480, 480)
    """

    targets = targets[:, None].float()    # (B, 1, 480, 480)
    loss_all = targets.new_zeros((1, ))

    for lno in range(len(preds)):
        loss_all = loss_all + F.binary_cross_entropy_with_logits(preds[lno],
                                                                 targets,
                                                                 reduction='mean',
                                                                 pos_weight=torch.FloatTensor([1.1]).cuda())

    return loss_all


def evaluate(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True), \
                                                   target.cuda(non_blocking=True), \
                                                   sentences.cuda(non_blocking=True), \
                                                   attentions.cuda(non_blocking=True)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])
                output = output[-1].cpu()
                output_mask = (output.sigmoid().squeeze(1) > 0.5).numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I * 1.0 / U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return cum_I * 100. / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq, iterations, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, sentences, text_mask=attentions)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_scheduler.step()

        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data

    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    dataset_train = get_dataset("train", get_transform(args=args), args=args, eval_mode=False)
    dataset_eval = get_dataset("val", get_transform(args=args), args=args, eval_mode=True)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_eval)

    # data loader
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
                                                    sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    model = lgformer.LGFormer(args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        {"params": reduce(operator.concat,
                          [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(12)])},
    ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(data_loader_train) * args.epochs)) ** 0.9)

    # native amp
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader_train, lr_scheduler, epoch, args.print_freq, iterations, scaler)

        overallIoU = evaluate(model, data_loader_eval)
        print('Overall IoU {}'.format(overallIoU))
        
        best_checkpoint = (best_oIoU < overallIoU)
        if best_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir, f'model_best_{args.model_id}.pth'))
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
