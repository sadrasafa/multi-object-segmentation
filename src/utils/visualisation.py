import torch
import torch.nn.functional as F
import torchvision
import functools
import collections
import warnings
import numpy as np
from depth_trainer import normalise_disparity, create_bases, project_flow_to_bases
import matplotlib.cm as cm

from cvbase.optflow.visualize import flow2rgb
from PIL import Image, ImageDraw, ImageFont

from .log import getLogger

LOGGER = getLogger(__name__)


def flow2rgb_torch(x):
    warnings.warn('Switch to using one from torchvision')
    return torch.from_numpy(flow2rgb(x.permute(1, 2, 0).numpy())).permute(2, 0, 1)


def create_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=np.int64)
    colormap[0] = [0, 0, 0]
    colormap[1] = [166, 206, 227]
    colormap[2] = [31, 120, 180]
    colormap[3] = [178, 223, 138]
    colormap[4] = [51, 160, 44]
    colormap[5] = [251, 154, 153]
    colormap[6] = [227, 26, 28]
    colormap[7] = [253, 191, 111]
    colormap[8] = [255, 127, 0]
    colormap[9] = [202, 178, 214]
    colormap[10] = [106, 61, 154]
    colormap[11] = [255, 255, 153]
    colormap[12] = [177, 89, 40]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return torch.from_numpy(colormap).long()

def __default_font(fontsize):
    try:
        FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", fontsize)
    except OSError:
        FNT = ImageFont.truetype("dejavu/DejaVuSans.ttf", fontsize)
    return FNT


@functools.lru_cache(None)  # cache the result
def autosized_default_font(size_limit: float) -> ImageFont.ImageFont:
    fontsize = 1  # starting font size
    font = __default_font(fontsize)
    while font.getsize('test123')[1] < size_limit:
        fontsize += 1
        font = __default_font(fontsize)
    fontsize -= 1
    font = __default_font(fontsize)
    return font

def get_vis_header(header_size, image_size, header_texts, header_height=20):
    W, H = (image_size, header_height)
    header_labels = []
    font = autosized_default_font(0.8 * H)

    for text in header_texts:
        im = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(im)
        w, h = draw.textsize(text, font=font)
        draw.text(((W - w) / 2, (H - h) / 2), text, fill="black", font=font)
        header_labels.append(torch.from_numpy(np.array(im)))
    header_labels = torch.cat(header_labels, dim=1)
    ret = (torch.ones((header_height, header_size, 3)) * 255)
    ret[:, :header_labels.size(1)] = header_labels

    return ret.permute(2, 0, 1).clip(0, 255).to(torch.uint8)


class Visualiser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = 32
        self.recon_flow = True

        self._cats = set()
        self._lbls = create_label_colormap().float()


        self._inputs = collections.defaultdict(list)
        self._extras = collections.defaultdict(list)

        self.res = cfg.UNSUPVIDSEG.RESOLUTION

    def add(self, sample_dict, disp, pred_dict, masks_softmaxed, pred_masks, true_masks=None):
        if len(self._cats) >= self.size:
            return
        category = sample_dict['category']
        frame_ids = sample_dict['frame_id']

        bases = create_bases(disp)  # 8xBx2xHxW
        m_softmaxed = torch.softmax(pred_dict['sem_seg'], dim=1) # BxKxHxW
    
        all_masked_bases = []
        for m_i in range(m_softmaxed.shape[1]):
            all_masked_bases.append([])
        for basis in bases:
            for m_i in range(m_softmaxed.shape[1]):
                all_masked_bases[m_i].append(m_softmaxed[:, m_i].unsqueeze(1) * basis)
        masked_bases = torch.cat([torch.stack(masked_bases_x, dim=-1) for masked_bases_x in all_masked_bases], dim=-1) #Bx2xHxWx(8*K)
        projected_flow = project_flow_to_bases(masked_bases, sample_dict['flow'].clip(-20, 20))
        if projected_flow is None:
            projected_flow = sample_dict['flow'].clip(-0.1, 0.1)

        for bi, (cat, fid) in enumerate(zip(category, frame_ids)):
            if cat not in self._cats:
                self._include(bi, sample_dict, disp, projected_flow, pred_dict, masks_softmaxed, pred_masks, true_masks)
                self._cats.add(cat)
            if len(self._cats) >= self.size:
                return

    def add_extra(self, sample_dict, col_name, tensor):
        if len(self._cats) >= self.size:
            return
        category = sample_dict['category']
        frame_ids = sample_dict['frame_id']
        for bi, (cat, fid) in enumerate(zip(category, frame_ids)):
            if cat in self._cats:
                self._extras[col_name].append(tensor[bi])


    def add_all(self, sample_dict, disp, pred_dict, masks_softmaxed, pred_masks, true_masks=None):
        if len(self._cats) >= self.size:
            return
        
        bases = create_bases(disp)  # 8xBx2xHxW
        m_softmaxed = torch.softmax(pred_dict['sem_seg'], dim=1) # BxKxHxW
    
        all_masked_bases = []
        for m_i in range(m_softmaxed.shape[1]):
            all_masked_bases.append([])
        for basis in bases:
            for m_i in range(m_softmaxed.shape[1]):
                all_masked_bases[m_i].append(m_softmaxed[:, m_i].unsqueeze(1) * basis)
        masked_bases = torch.cat([torch.stack(masked_bases_x, dim=-1) for masked_bases_x in all_masked_bases], dim=-1) #Bx2xHxWx(8*K)
        projected_flow = project_flow_to_bases(masked_bases, sample_dict['flow'].clip(-20, 20))
        if projected_flow is None:
            projected_flow = sample_dict['flow'].clip(-0.1, 0.1)

        for bi in range(pred_masks.shape[0]):
            self._include(bi, sample_dict, disp, projected_flow, pred_dict, masks_softmaxed, pred_masks, true_masks)
            self._cats.add(len(self._cats))
            if len(self._cats) >= self.size:
                return

    def _include(self, i, sample_dict, disp, projected_flow, pred_dict, masks_softmaxed, pred_masks, true_masks=None):
        # RGB / input
        LOGGER.debug_once(f'Adding input {sample_dict[self.cfg.UNSUPVIDSEG.SAMPLE_KEYS[0]].shape}')
        self._inputs['input'].append(sample_dict[self.cfg.UNSUPVIDSEG.SAMPLE_KEYS[0]][i].detach())
        # Disp
        LOGGER.debug_once(f'Adding disparity {disp.shape}')
        self._inputs['disp'].append(disp[i].detach())
        # Projected Flow
        LOGGER.debug_once(f'Adding projected flow {projected_flow.shape}')
        self._inputs['projected_flow'].append(projected_flow[i].clamp(-20, 20).detach())
        # Flow
        LOGGER.debug_once(f'Adding flow {sample_dict["flow"].shape}')
        self._inputs['flow'].append(sample_dict['flow'][i].clamp(-20, 20).detach())
        # Masks
        if true_masks is not None:
            LOGGER.debug_once(f'Adding true masks {true_masks.shape}')
            self._inputs['mask_true'].append(true_masks[i].detach())
        LOGGER.debug_once(f'Adding pred masks {pred_masks.shape}')
        self._inputs['mask_pred'].append(pred_masks[i].detach())
        # Slots
        LOGGER.debug_once(f'Adding slots {masks_softmaxed.shape}')
        self._inputs['slots'].append(masks_softmaxed[i])

    def _format(self, img, name='<unspecified_img>', shape_check=True):
        if img.max() > 1.0:
            # Assuming float in 0 - 255 range
            img = img.clamp(0., 255.).to(torch.uint8)
        else:
            img = (img.clamp(0, 1) * 255).to(torch.uint8)

        if len(img.shape) == 2:
            img = img[None]
        if len(img.shape) == 3:
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            img = img[:3]
        else:
            raise ValueError(f"Unkown img shape {img.shape}")
        if shape_check and img.shape[-2:] != self.res:
            LOGGER.debug_once(f'Need to interpolate {name} for vis {img.shape[-2:]} -> {self.res}')
            img = F.interpolate(img[None].float(), size=self.res, mode='bilinear', align_corners=False)[0].to(img.dtype)
        return img

    def _cmap(self, img, cmap):
        raise NotImplementedError

    def _seg(self, img):
        return torch.einsum('khw,kc->chw', img.squeeze(1).cpu(), self._lbls[:img.shape[0]].to(img.dtype))

    @functools.cached_property
    def _flow(self):
        _ = self._inputs['flow'][0]
        return torch.stack([f.to(_.device).to(_.dtype) for f in self._inputs['flow']])

    @functools.cached_property
    def _mask(self):
        _ = self._inputs['flow'][0]
        return torch.stack([m.to(_.device).to(_.dtype) for m in self._inputs['slots']])

    def add_col(self, col_name, tensor):
        self._extras[col_name] = [t for t in tensor]

    def img_vis(self):
        cols = [
            [self._format(x, 'input').cpu() for x in self._inputs['input']],
            [self._format(viz_disp(x.unsqueeze(0).cpu()).squeeze(0), 'disp') for x in self._inputs['disp']],
            [self._format(torchvision.utils.flow_to_image(x.cpu()), 'projected flow') for x in self._inputs['projected_flow']],
            [self._format(torchvision.utils.flow_to_image(x.cpu()), 'flow') for x in self._inputs['flow']],
        ]
        head = ['input', 'disp', 'projected_flow', 'gt_flow']
        if 'mask_true' in self._inputs:
            cols.append([self._format(self._seg(x.cpu()), 'mt') for x in self._inputs['mask_true']])
            head.append('gt_seg')
        cols.append([self._format(self._seg(x.cpu()), 'mp') for x in self._inputs['mask_pred']])
        head.append('pred_seg')

        for col_name in self._extras:
            cols.append([self._format(x.cpu(), 'e_'+col_name) for x in self._extras[col_name]])
            head.append(col_name)

        for k in range(self._inputs['slots'][0].shape[0]):
            cols.append([self._format(x[k].cpu(), f'slot_{k}') for x in self._inputs['slots']])
            head.append(f'slot_{k}')

        # Into grid
        vis = []
        for r in range(len(cols[0])):
            for c in range(len(cols)):
                vis.append(cols[c][r])
        vis = torchvision.utils.make_grid(vis, nrow=len(cols), pad=2)
        hed = self._format(get_vis_header(vis.shape[-1], cols[0][0].shape[-1] + 2, head), 'header', shape_check=False)
        return torch.cat([hed, vis], dim=1)


def batch_grid(imgs):
    B, K, *C, H, W = imgs.shape
    c = C[0] if C else 1
    if len(C) > 1:
        raise ValueError
    imgs = imgs.view(B*K, c, H, W).detach().cpu()
    return torchvision.utils.make_grid(imgs, nrow=K, pad=2)

def viz_disp_simple(disp):
    normalised_disp = normalise_disparity(disp).squeeze(1).detach().cpu()
    return (torch.stack([torch.stack([d, d, d]) for d in normalised_disp]) * 255).clip(0, 255).to(torch.uint8)

def viz_disp(disp):
    mapper = cm.ScalarMappable(norm=None, cmap='magma')
    normalised_disp = normalise_disparity(disp).squeeze(1).detach().cpu().numpy()
    return (torch.stack([torch.from_numpy(mapper.to_rgba(d)[:, :, :3]) for d in normalised_disp]).permute(0, 3, 1, 2) * 255).clip(0, 255).to(torch.uint8)