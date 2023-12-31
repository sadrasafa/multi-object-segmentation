# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import sys
import itertools
import logging
import os
# import wandb
import json

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

from detectron2.utils.file_io import PathManager

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_maskformer2_video_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)

from config import add_unsup_vidseg_config

from pathlib import Path

import utils

logger = logging.getLogger('unsup_vidseg')


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_depth_model(cls, cfg):
        assert cfg.UNSUPVIDSEG.DEPTH.MODEL in ["MiDaS", "DPT_Large", "DPT_SwinV2_B_384", "DPT_SwinV2_L_384"]
        depth_model = torch.hub.load("intel-isl/MiDaS", cfg.UNSUPVIDSEG.DEPTH.MODEL, pretrained=False)
        depth_model.to(torch.device(cfg.MODEL.DEVICE))
        return depth_model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return YTVISEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        mapper = YTVISDatasetMapper(cfg, is_train=True)

        dataset_dict = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_depth_optimizer(cls, cfg, depth_model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        depth_defaults = {}
        depth_defaults["lr"] = cfg.UNSUPVIDSEG.DEPTH.LR
        depth_defaults["weight_decay"] = cfg.UNSUPVIDSEG.DEPTH.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        depth_params: List[Dict[str, Any]] = []
        depth_memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in depth_model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in depth_memo:
                    continue
                depth_memo.add(value)
                hyperparams = copy.copy(depth_defaults)
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                depth_params.append({"params": [value], **hyperparams})

        def maybe_add_full_depth_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
                    and cfg.UNSUPVIDSEG.DEPTH.CLIP_GRADIENTS
            )

            class FullDepthModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullDepthModelGradientClippingOptimizer if enable else optim

        depth_optimizer_type = cfg.UNSUPVIDSEG.DEPTH.OPTIMIZER
        if depth_optimizer_type == "SGD":
            depth_optimizer = maybe_add_full_depth_model_gradient_clipping(torch.optim.SGD)(
                depth_params, cfg.UNSUPVIDSEG.DEPTH.LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif depth_optimizer_type == "ADAMW":
            depth_optimizer = maybe_add_full_depth_model_gradient_clipping(torch.optim.AdamW)(
                depth_params, cfg.UNSUPVIDSEG.DEPTH.LR
            )
        elif depth_optimizer_type == "RMSProp":
            depth_optimizer = maybe_add_full_depth_model_gradient_clipping(torch.optim.RMSprop)(
                depth_params, cfg.UNSUPVIDSEG.DEPTH.LR
            )
        elif depth_optimizer_type == "ADAM":
            depth_optimizer = maybe_add_full_depth_model_gradient_clipping(torch.optim.Adam)(
                depth_params, cfg.UNSUPVIDSEG.DEPTH.LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {depth_optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            depth_optimizer = maybe_add_gradient_clipping(cfg, depth_optimizer)


        return depth_optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former_video")
    return cfg



def setup(args):
    """
    Create configs and perform basic setups.
    """

    should_resume = args.resume or args.resume_path is not None

    if 'CONFIG_FILE' in args.opts:
        logger.warning(
            f"Found CONFIG_FILE key in OPT args and using {args.opts[args.opts.index('CONFIG_FILE') + 1]} instead of {args.config_file}")
        args.config_file = args.opts[args.opts.index('CONFIG_FILE') + 1]
    

    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)

    add_unsup_vidseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    datestring = utils.log.get_datestring_for_the_run()
    
    if should_resume:
        if args.resume_path is None:
            path = Path(cfg.OUTPUT_BASEDIR) / cfg.LOG_ID

            checkpoints = sorted([i for i in path.rglob('checkpoints/*.pth')], key=lambda p: p.lstat().st_mtime)
            print(checkpoints)
            args.resume_path = str(checkpoints[-1])


        cfg.OUTPUT_DIR = "/".join(args.resume_path.split('/')[:-2])  # LOG_ID/datestring/checkpoints/checkpoints.pth

        if args.eval_only:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'eval', datestring)

    else:
        if cfg.LOG_ID:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_BASEDIR, cfg.LOG_ID)
        else:
            cfg.OUTPUT_DIR = cfg.OUTPUT_BASEDIR

        if args.eval_only:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'eval', datestring)
        else:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datestring)
            os.makedirs(f'{cfg.OUTPUT_DIR}/checkpoints', exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger for "unsup_vidseg" module
    setup_logger(output=f'{cfg.OUTPUT_DIR}/main.log', distributed_rank=comm.get_rank(), name="unsup_vidseg")
    with open(f'{cfg.OUTPUT_DIR}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return cfg
