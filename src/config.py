import copy
import glob as gb
import itertools
import os
from functools import partial
from pathlib import Path

import numpy as np
import numpy.random
import torch.utils.data
from detectron2.config import CfgNode as CN

import utils
from datasets import FlowPairCMDetectron, \
    FlowPairMoviDetectron, KITTI_VAL, KITTI_Train, FlowPairDetectron, FlowEvalDetectron

logger = utils.log.getLogger('unsup_vidseg')


def setup_movi_dataset(cfg, num_frames=1):
    cache_path = None
    resolution = cfg.UNSUPVIDSEG.RESOLUTION  # h,w
    prefix = f'data/{cfg.UNSUPVIDSEG.DATASET.lower()}'

    train_dataset = FlowPairMoviDetectron(
        'train', None,
        resolution,
        prefix=prefix,
        gt_flow=True,
        cache_path=cache_path,
        num_frames=num_frames,
        two_flow=num_frames > 1 or cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW
    )
    val_dataset = FlowPairMoviDetectron(
        'validation',
        None,
        resolution,
        prefix=prefix,
        gt_flow=True,
        cache_path=cache_path,
        num_frames=None if cfg.EVAL_WHOLE_SEQ else 1
        )
    return train_dataset, val_dataset


def setup_moving_clevrtex(cfg, num_frames=1):
    pairs = [1, 2, -1, -2]

    cache_path = None
    resolution = cfg.UNSUPVIDSEG.RESOLUTION  # h,w

    if cfg.UNSUPVIDSEG.DATASET in ['CM.M', 'CM.M.GT', 'CM.M.F.GT']:

        prefix = 'data/moving_clevrtex'

    elif cfg.UNSUPVIDSEG.DATASET in ['CM.R', 'CM.R.GT', 'CM.R.F.GT']:

        prefix = 'data/moving_clevr'

    if cache_path is None:
        logger.warn("Cache path is not set, caching will be disabled.")

    train_dataset = FlowPairCMDetectron(
        split='train',  # This will sample
        pairs=pairs,
        flow_dir=None,
        res='240p',
        resolution=resolution,
        to_rgb=False,
        size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
        first=None,
        prefix=prefix,
        gt_flow=True,
        with_clevr=False,
        single_sample=False,
        two_flow=num_frames > 1 or cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW,
        ccrop=True,
        cache_path=cache_path,
        filter=False,
        num_frames=num_frames,
        no_lims=True,
        first_frame_only=False,
        darken=False
    )
    val_dataset = FlowPairCMDetectron(
        split='val',  # This will process sequentially
        pairs=pairs,  # only first "flow pair" will be used
        flow_dir=None,
        res='240p',
        resolution=resolution,
        to_rgb=False,
        single_sample=False,
        size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
        first=None,
        prefix=prefix,
        gt_flow=True,
        with_clevr=False,
        two_flow=False,
        ccrop=True,
        cache_path=cache_path,
        filter=False,
        num_frames=None if cfg.EVAL_WHOLE_SEQ else 1,
        no_lims=True,
        darken=False
    )

    return train_dataset, val_dataset

def setup_kitti_dataset(cfg):
    cache_path = None
    if not isinstance(cache_path, (str, Path)):
        cache_path = None

    if cfg.FLAGS.NO_CACHE:
        cache_path = None

    train_dataset = KITTI_Train('data/KITTI/KITTI-Raw', 'data/KITTI/RAFT_FLOWS', cfg.UNSUPVIDSEG.RESOLUTION)

    val_dataset = KITTI_VAL(cfg.UNSUPVIDSEG.RESOLUTION,
                            prefix='data/KITTI/KITTI_test',
                            cache_path=cache_path,
                            size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                            return_original=True,)
    return train_dataset, val_dataset

def setup_davis17_dataset(cfg):
    resolution = (128, 224)
    pairs = [4, 8, -4, -8]
    basepath = '/DAVIS2017'
    img_dir = '/DAVIS2017/JPEGImages/480p'
    if cfg.UNSUPVIDSEG.DEPTH.DAVIS_ANNO == "motion":
        gt_dir = '/DAVIS2017/Annotations_unsupervised_motion/480p'
    elif cfg.UNSUPVIDSEG.DEPTH.DAVIS_ANNO == "unsupervised":
        gt_dir = '/DAVIS2017/Annotations_unsupervised/480p'
    else:
        gt_dir = '/DAVIS2017/Annotations/480p'

    val_flow_dir = '/DAVIS2017/Flows_gap4'
    val_seq = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 
                'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
                'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick', 'motocross-jump', 
                'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
    val_data_dir = [val_flow_dir, img_dir, gt_dir]
    res = ""

    root_path_str = 'data'

    if root_path_str.startswith('/'):
        root_path = Path(f"/{root_path_str.lstrip('/').rstrip('/')}")
    else:
        root_path = Path(f"{root_path_str.lstrip('/').rstrip('/')}")

    logger.info(f"Loading dataset from: {root_path}")

    basepath = root_path / basepath.lstrip('/').rstrip('/')
    img_dir = root_path / img_dir.lstrip('/').rstrip('/')
    gt_dir = root_path / gt_dir.lstrip('/').rstrip('/')
    val_data_dir = [root_path / path.lstrip('/').rstrip('/') for path in val_data_dir]

    folders = [p.name for p in (basepath / f'Flows_gap{pairs[0]}').iterdir() if p.is_dir()]
    folders = sorted(folders)

    # flow_dir is a dictionary, with keys indicating the flow gap, and each value is a list of sequence names,
    # each item then is an array with Nx2, N indicates the number of available pairs.

    flow_dir = scan_train_flow(folders, res, pairs, basepath)
    data_dir = [flow_dir, img_dir, gt_dir]

    # force1080p = ('DAVIS' not in cfg.GWM.DATASET) and 'RGB_BIG' in cfg.GWM.SAMPLE_KEYS
    force1080p = False

    # enable_photometric_augmentations = cfg.FLAGS.INF_TPS
    enable_photometric_augmentations = False

    train_dataset = FlowPairDetectron(data_dir=data_dir,
                                    resolution=resolution,
                                    to_rgb=False,
                                    size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                                    enable_photo_aug=enable_photometric_augmentations,
                                    flow_clip=cfg.UNSUPVIDSEG.FLOW_LIM,
                                    norm=False,
                                    force1080p=force1080p,
                                    flow_res=None,)
    val_dataset = FlowEvalDetectron(data_dir=val_data_dir,
                                    resolution=resolution,
                                    pair_list=pairs,
                                    val_seq=val_seq,
                                    to_rgb=False,
                                    with_rgb=False,
                                    size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                                    flow_clip=cfg.UNSUPVIDSEG.FLOW_LIM,
                                    norm=False,
                                    force1080p=force1080p)
    return train_dataset, val_dataset


def setup_dataset(cfg=None, num_frames=1):
    dataset_str = cfg.UNSUPVIDSEG.DATASET
    if '+' in dataset_str:
        datasets = dataset_str.split('+')
        logger.info(f'Multiple datasets detected: {datasets}')
        train_datasets = []
        val_datasets = []
        val_dataset_index = 0
        for i, ds in enumerate(datasets):
            if ds.startswith('val(') and ds.endswith(')'):
                val_dataset_index = i
                ds = ds[4:-1]
            proxy_cfg = copy.deepcopy(cfg)
            proxy_cfg.merge_from_list(['UNSUPVIDSEG.DATASET', ds]),
            train_ds, val_ds = setup_dataset(proxy_cfg)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        logger.info(f'Multiple datasets detected: {datasets}')
        logger.info(f'Validation is still : {datasets[val_dataset_index]}')
        return torch.utils.data.ConcatDataset(train_datasets), val_datasets[val_dataset_index]

    
    if cfg.UNSUPVIDSEG.DATASET.startswith('MOVi'):
        return setup_movi_dataset(cfg, num_frames)

    if cfg.UNSUPVIDSEG.DATASET.startswith('CM'):
        return setup_moving_clevrtex(cfg, num_frames)


    if cfg.UNSUPVIDSEG.DATASET in ['KITTIDATASET']:
        return setup_kitti_dataset(cfg)
    

    if cfg.UNSUPVIDSEG.DATASET in ['DAVIS2017']:
        return setup_davis17_dataset(cfg)

    raise ValueError(f'Unknown dataset {cfg.UNSUPVIDSEG.DATASET}')


def loaders(cfg, video=False):
    nf = 1
    if video:
        nf = cfg.INPUT.SAMPLING_FRAME_NUM
    train_dataset, val_dataset = setup_dataset(cfg, num_frames=nf)
    logger.info(f"Sourcing data from {val_dataset.data_dir[0]}")
    logger.info(f"training dataset: {train_dataset}")
    logger.info(f"val dataset: {train_dataset}")

    if cfg.FLAGS.DEV_DATA:
        subset = cfg.SOLVER.IMS_PER_BATCH * 3
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))

    g = torch.Generator()
    data_generator_seed = int(torch.randint(int(1e6), (1,)).item())
    logger.info(f"Dataloaders generator seed {data_generator_seed}")
    g.manual_seed(data_generator_seed)

    val_loader_size = 1
    if not cfg.EVAL_WHOLE_SEQ and (cfg.UNSUPVIDSEG.DATASET.startswith('CM')
                                   or cfg.UNSUPVIDSEG.DATASET.startswith('MOVi')):
        val_loader_size = max(cfg.SOLVER.IMS_PER_BATCH, 16)
        logger.info(f"Increasing val loader size to {val_loader_size}")
    if cfg.UNSUPVIDSEG.DATASET in ['KITTIDATASET', 'DAVIS2017']:
        val_loader_size = 1  # Enfore singe-sample val for KITTI

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                               collate_fn=lambda x: x,
                                               shuffle=True,
                                               pin_memory=False,
                                               drop_last=True,
                                               persistent_workers=False and cfg.DATALOADER.NUM_WORKERS > 0,
                                               worker_init_fn=utils.random_state.worker_init_function,
                                               generator=g
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             batch_size=val_loader_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             collate_fn=lambda x: x,
                                             drop_last=False,
                                             persistent_workers=False and cfg.DATALOADER.NUM_WORKERS > 0,
                                             worker_init_fn=utils.random_state.worker_init_function,
                                             generator=g)
    if cfg.FLAGS.TRAINVAL:
        rng = np.random.default_rng(seed=42)
        train_dataset_clone = copy.deepcopy(train_dataset)
        train_dataset_clone.dataset.random = False
        trainval_dataset = torch.utils.data.Subset(train_dataset_clone,
                                                   rng.choice(len(train_dataset), len(val_dataset), replace=False))
        trainval_loader = torch.utils.data.DataLoader(trainval_dataset,
                                                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                      batch_size=val_loader_size,
                                                      shuffle=False,
                                                      pin_memory=False,
                                                      collate_fn=lambda x: x,
                                                      drop_last=False,
                                                      persistent_workers=False and cfg.DATALOADER.NUM_WORKERS > 0,
                                                      worker_init_fn=utils.random_state.worker_init_function,
                                                      generator=g)
        return train_loader, trainval_loader, val_loader
    return train_loader, val_loader


def add_unsup_vidseg_config(cfg):
    cfg.UNSUPVIDSEG = CN()
    cfg.UNSUPVIDSEG.RESOLUTION = (128, 128)
    cfg.UNSUPVIDSEG.SAMPLE_KEYS = ["rgb"]
    cfg.UNSUPVIDSEG.DATASET = 'CM.M.GT'
    cfg.UNSUPVIDSEG.USE_MULT_FLOW = False

    cfg.UNSUPVIDSEG.DEPTH = CN()
    cfg.UNSUPVIDSEG.DEPTH.MODEL = "MiDaS"
    cfg.UNSUPVIDSEG.DEPTH.OPTIMIZER = "ADAM"
    cfg.UNSUPVIDSEG.DEPTH.LR = 5e-5
    cfg.UNSUPVIDSEG.DEPTH.WEIGHT_DECAY = 1e-6
    cfg.UNSUPVIDSEG.DEPTH.CLIP_GRADIENTS = False
    cfg.UNSUPVIDSEG.DEPTH.FLOW_TYPE = "RAFT"
    cfg.UNSUPVIDSEG.DEPTH.DAVIS_ANNO = "motion"


    cfg.UNSUPVIDSEG.FLOW_LIM = 20


    cfg.UNSUPVIDSEG.LOSS = 'ELBO_AFF_FULL'
    cfg.UNSUPVIDSEG.LOSS_ORIGIN = 'centroid_fix'
    cfg.UNSUPVIDSEG.LOSS_FLOW_KEY = 'flow'


    cfg.UNSUPVIDSEG.LOSS_GRID_DETACH = False
    cfg.UNSUPVIDSEG.LOSS_BETA = 'lin(5000,0.1,-0.1)'
    cfg.UNSUPVIDSEG.LOSS_NPART = 3
    cfg.UNSUPVIDSEG.LOSS_TEMP = 'const(1.0)'
    cfg.UNSUPVIDSEG.LOSS_SIGMA2 = 'const(0.5)'


    cfg.UNSUPVIDSEG.LOSS_COV = 'simple'
    cfg.UNSUPVIDSEG.LOSS_MEANS = False


    cfg.UNSUPVIDSEG.LOSS_EQUIV = 'const(0.0)'
    cfg.UNSUPVIDSEG.LOSS_DISP_THRESH = -1.0
    cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW = False


    cfg.FLAGS = CN()
    cfg.FLAGS.METRIC = 'mIOU'
    cfg.FLAGS.KEEP_ALL = False  # Keep all checkoints

    cfg.FLAGS.UNFREEZE_AT = []

    cfg.FLAGS.DEV_DATA = False  # Run with artificially downsampled dataset for fast dev
    cfg.FLAGS.USE_CCPP = True
    
    cfg.FLAGS.GC_FREQ = 10
    cfg.FLAGS.TRAINVAL = False

    cfg.FLAGS.NO_CACHE = False

    cfg.DEBUG = False

    cfg.LOG_ID = 'exp'
    cfg.LOG_FREQ = 1000
    cfg.OUTPUT_BASEDIR = '../outputs'
    cfg.TOTAL_ITER = None
    cfg.CONFIG_FILE = None

    cfg.EVAL_WHOLE_SEQ = False
    

    if os.environ.get('SLURM_JOB_ID', None):
        cfg.LOG_ID = os.environ.get('SLURM_JOB_NAME', cfg.LOG_ID)
        print(f"Setting name {cfg.LOG_ID} based on SLURM job name")

def scan_train_flow(folders, res, pairs, basepath):
    pair_list = [p for p in itertools.combinations(pairs, 2)]

    flow_dir = {}
    for pair in pair_list:
        p1, p2 = pair
        flowpairs = []
        for f in folders:
            path1 = basepath / f'Flows_gap{p1}' / res / f
            path2 = basepath / f'Flows_gap{p2}' / res / f

            flows1 = [p.name for p in path1.glob('*.flo')]
            flows2 = [p.name for p in path2.glob('*.flo')]

            flows1 = sorted(flows1)
            flows2 = sorted(flows2)

            intersect = list(set(flows1).intersection(flows2))
            intersect.sort()

            flowpair = np.array([[path1 / i, path2 / i] for i in intersect])
            flowpairs += [flowpair]
        flow_dir['gap_{}_{}'.format(p1, p2)] = flowpairs

    # flow_dir is a dictionary, with keys indicating the flow gap, and each value is a list of sequence names,
    # each item then is an array with Nx2, N indicates the number of available pairs.
    return flow_dir