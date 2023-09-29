import determinism  # noqa

determinism.do_not_delete()  # noqa

import argparse
import os
import sys
import time
from argparse import ArgumentParser
import gc

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import config
import val
from mask2former_trainer_video import setup, Trainer
from depth_trainer import  create_bases, project_flow_to_bases, normalize_rgb


# @formatter:on

logger = utils.log.getLogger('unsup_vidseg')


torch.multiprocessing.set_sharing_strategy('file_system')


def train_step(cfg, model, depth_model, optimizer, depth_optimizer, scheduler, sample, iteration, total_iter):
    sample_dict = utils.convert.list_of_dicts_2_dict_of_tensors(sample, device=model.device)
    logger.debug_once(f'Train inputs: {[(k, utils.get_shape(v)) for k, v in sample_dict.items()]}')

    preds = model.forward_base(sample, keys=cfg.UNSUPVIDSEG.SAMPLE_KEYS, get_eval=True)

    preds_dict = utils.convert.list_of_dicts_2_dict_of_tensors(preds, device=model.device)
    logger.debug_once(f'Train outputs: {[(k, utils.get_shape(v)) for k,v in preds_dict.items()]}')

    logger.debug_once(f'Using single flow mode')

    flow = sample_dict['flow'].clip(-cfg.UNSUPVIDSEG.FLOW_LIM, cfg.UNSUPVIDSEG.FLOW_LIM)

    masks_softmaxed = torch.softmax(preds_dict['sem_seg'], dim=1) # BxKxHxW

    rgb = normalize_rgb(sample, cfg.UNSUPVIDSEG.DEPTH.MODEL).to(model.device)
    disp = F.interpolate(depth_model(rgb).unsqueeze(1),  masks_softmaxed.shape[-2:])    # Bx1xHxW                
    bases = create_bases(disp)  # 8xBx2xHxW
    
    all_masked_bases = []
    for m_i in range(masks_softmaxed.shape[1]):
        all_masked_bases.append([])
    for basis in bases:
        for m_i in range(masks_softmaxed.shape[1]):
            all_masked_bases[m_i].append(masks_softmaxed[:, m_i].unsqueeze(1) * basis)
    masked_bases = torch.cat([torch.stack(masked_bases_x, dim=-1) for masked_bases_x in all_masked_bases], dim=-1) #Bx2xHxWx(8*K)
    
    projected_flow = project_flow_to_bases(masked_bases, flow)
    if projected_flow is None:
        return -1, {f'train/loss_depth': -1}
    loss_depth = torch.linalg.norm(flow - projected_flow, dim=1).mean()
    loss = loss_depth
    
    train_log_dict = {f'train/loss_depth': loss_depth}
    train_log_dict['train/loss'] = loss
    train_log_dict['train/learning_rate'] = optimizer.param_groups[-1]['lr']
    train_log_dict['train/learning_rate_depth'] = depth_optimizer.param_groups[-1]['lr']

    optimizer.zero_grad()
    depth_optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    scheduler.step()
    depth_optimizer.step()
    return loss.item(), train_log_dict


def train_vis(cfg, model, depth_model, sample):
    epreds = model.forward_base(sample, keys=cfg.UNSUPVIDSEG.SAMPLE_KEYS, get_eval=True)
    epreds_dict = utils.convert.list_of_dicts_2_dict_of_tensors(epreds, device=model.device)
    sample_dict = utils.convert.list_of_dicts_2_dict_of_tensors(sample, device=model.device)
    if cfg.INPUT.SAMPLING_FRAME_NUM > 1:
        sample_dict = utils.convert.to_batchxtime(sample_dict)
    masks_softmaxed, pred_masks, true_masks = val.get_masks(cfg, epreds_dict, sample_dict)
    rgb = normalize_rgb(sample, cfg.UNSUPVIDSEG.DEPTH.MODEL).to(model.device)
    disp = F.interpolate(depth_model(rgb).unsqueeze(1),  epreds_dict['sem_seg'].shape[-2:])    # Bx1xHxW                
    vis = utils.visualisation.Visualiser(cfg)
    vis.add_all(sample_dict, disp, epreds_dict, masks_softmaxed, pred_masks, true_masks)
    imgs = [vis.img_vis()]
    return imgs

def main(args):
    cfg = setup(args)
    logger.info(f"Called as {' '.join(sys.argv)}")
    logger.info(f'Output dir {cfg.OUTPUT_DIR}')

    random_state = utils.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))
    random_state.seed_everything()
    utils.log.checkpoint_code(cfg.OUTPUT_DIR)

    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    # initialize model
    model = Trainer.build_model(cfg)
    # print(model.sem_seg_head.predictor.whole_seq)
    model.sem_seg_head.predictor.whole_seq = cfg.EVAL_WHOLE_SEQ

    logger.info('Checking backbone trainability')
    if hasattr(model, 'backbone'):
        for n, p in model.backbone.named_parameters():
            if not p.requires_grad:
                logger.warning(f'{n} is not trainable in backbone')
    else:
        logger.warning('model.backbone not found')

    optimizer = Trainer.build_optimizer(cfg, model)
    scheduler = Trainer.build_lr_scheduler(cfg, optimizer)
    scheduler.max_iters = cfg.SOLVER.MAX_ITER  # Reset if config changed

    depth_model = Trainer.build_depth_model(cfg)
    depth_optimizer = Trainer.build_depth_optimizer(cfg, depth_model)

    logger.info(f'Optimiser is {type(optimizer)}')
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params {pytorch_total_params} (train) {pytorch_total_train_params}')


    checkpointer = DetectionCheckpointer(model,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, 'checkpoints'),
                                         random_state=random_state,
                                         optimizer=optimizer,
                                         scheduler=scheduler)
    checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume_path is not None)

    if cfg.MODEL.WEIGHTS is not None:
        depth_ckpt = torch.load(cfg.MODEL.WEIGHTS.replace('_seg.pth', '_depth.pth'))
        depth_model.load_state_dict(depth_ckpt['depth_model'])
        depth_optimizer.load_state_dict(depth_ckpt['depth_optimizer'])

    iteration = 0 if args.resume_path is None else checkpoint['iteration']

    train_loader, val_loader = config.loaders(cfg, video=True)


    if args.eval_only:
        if len(val_loader.dataset) == 0:
            logger.error("Training dataset: empty")
            sys.exit(0)
        model.eval()
        depth_model.eval()
        metrics = val.run_evaluation(cfg=cfg, val_loader=val_loader, model=model, depth_model=depth_model, writer=writer, writer_iteration=iteration, video=True)
        print(metrics)
        return

    if len(train_loader.dataset) == 0:
        logger.error("Training dataset: empty")
        sys.exit(0)

    logger.info(
        f'Start of training: dataset {cfg.UNSUPVIDSEG.DATASET},'
        f' train {len(train_loader.dataset)}, val {len(val_loader.dataset)},'
        f' device {model.device}, keys {cfg.UNSUPVIDSEG.SAMPLE_KEYS}, '
        f'multiple flows {cfg.UNSUPVIDSEG.USE_MULT_FLOW}')

    iou_best = 0
    ARI_F_best = 0
    timestart = time.time()

    total_iter = cfg.TOTAL_ITER if cfg.TOTAL_ITER else cfg.SOLVER.MAX_ITER  # early stop


    gc_hist = []

    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and \
         tqdm(initial=iteration, total=total_iter, disable=False) as pbar:
        while iteration < total_iter:
            for sample in train_loader:
                sample = [e for s in sample for e in s]
                logger.info_once(f"RGB: {sample[0]['rgb'].shape} {sample[0]['flow'].shape} {sample[0]['sem_seg'].shape}")

                loss, train_log_dict = train_step(cfg, model, depth_model, optimizer, depth_optimizer, scheduler, sample, iteration, total_iter)

                pbar.set_postfix(loss=loss)
                pbar.update()

                if (iteration + 1) % cfg.FLAGS.GC_FREQ == 0 or iteration < 100:
                    gc_hist.append(gc.collect())
                    gc_hist = gc_hist[-100:]

                if (iteration + 1) % 1000 == 0 or iteration + 1 in {1, 50}:
                    logger.info(
                        f'Iteration {iteration + 1}. AVG GC {np.nanmean(gc_hist)}. RNG outputs {utils.random_state.get_randstate_magic_numbers(model.device)}')

                if cfg.DEBUG or (iteration + 1) % 100 == 0:
                    logger.info(
                        f'Iteration: {iteration + 1}, time: {time.time() - timestart:.01f}s, loss: {loss:.02f}.')

                    for k, v in train_log_dict.items():
                        if writer:
                            writer.add_scalar(k, v, iteration + 1)

                    if writer:
                        writer.add_scalar('util/train_max_gpu_mem', torch.cuda.max_memory_allocated() / 2.0**20, iteration + 1)
                        torch.cuda.reset_max_memory_allocated()


                iteration += 1
                timestart = time.time()
                if iteration >= total_iter:
                    logger.info("Stopping")
                    checkpointer.save(name='checkpoint_final', iteration=iteration, loss=loss,
                                      iou=iou_best)
                    torch.save({'iteration': iteration, 'depth_model': depth_model.state_dict(),
                                            'depth_optimizer': depth_optimizer.state_dict(),
                                            'loss': loss}, os.path.join(cfg.OUTPUT_DIR, 'checkpoints', 'checkpoint_last_depth.pth'))
                    return iou_best  # Done
                del train_log_dict

                if (iteration) % cfg.LOG_FREQ == 0 or (iteration) in [50, 500]:
                    logger.info(f"Eval GC: {gc.collect()}")
                    model.eval()
                    depth_model.eval()
                    torch.cuda.reset_max_memory_allocated()
                    if writer:
                        with torch.no_grad():
                            image_viz = train_vis(cfg, model, depth_model, sample)
                            writer.add_image('train/images', image_viz[0], iteration)
                            if len(image_viz) > 1:
                                for i in range(1, len(image_viz)):
                                    writer.add_image(f'extras/train_{i}', image_viz[i], iteration)

                    if metrics := val.run_evaluation(cfg=cfg,
                                                    val_loader=val_loader,
                                                    model=model,
                                                    depth_model=depth_model,
                                                    writer=writer,
                                                    writer_iteration=iteration,
                                                    video=True):
                        
                        if writer:
                            writer.add_scalar('util/eval_max_gpu_mem', torch.cuda.max_memory_allocated() / 2.0**20, iteration)
                    
                    iou_this, ARI_F_this = metrics['mIoU'], metrics['ARI-F']
                    if iou_this > iou_best:
                        iou_best = iou_this
                        checkpointer.save(name='checkpoint_best_iou', iteration=iteration, loss=loss,
                                      iou=iou_best)
                        torch.save({'iteration': iteration, 'depth_model': depth_model.state_dict(),
                                                'depth_optimizer': depth_optimizer.state_dict(),
                                                'loss': loss}, os.path.join(cfg.OUTPUT_DIR, 'checkpoints', 'checkpoint_best_iou_depth.pth'))
                    if ARI_F_this > ARI_F_best:
                        ARI_F_best = ARI_F_this
                        checkpointer.save(name='checkpoint_best_ARI_F', iteration=iteration, loss=loss,
                                      iou=iou_best)
                        torch.save({'iteration': iteration, 'depth_model': depth_model.state_dict(),
                                                'depth_optimizer': depth_optimizer.state_dict(),
                                                'loss': loss}, os.path.join(cfg.OUTPUT_DIR, 'checkpoints', 'checkpoint_best_ARI_F_depth.pth'))
                    
                    model.train()
                    depth_model.train()

def get_argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--config-file', type=str,
                        default='configs/mask2former/swin/maskformer2_swin_tiny_bs16_160k.yaml')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_argparse_args().parse_args()
    if args.resume_path:
        args.config_file = "/".join(args.resume_path.split('/')[:-2]) + '/config.yaml'
        print(args.config_file)
    main(args)
