# import numpy as np
import torch
# import torch.nn
import torch.nn.functional as F
from torchvision import transforms

def normalize_imagenet(rgb):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform(rgb / 255)

def normalize_rgb(sample, depth_model_type):
    if depth_model_type == 'MiDaS':
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb = torch.stack([x['rgb'] for x in sample])
        return transform(rgb / 255)
    elif depth_model_type == 'DPT_Large':
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        rgb = torch.stack([x['rgb'] for x in sample])
        return transform(rgb / 255)
    elif depth_model_type in ['DPT_SwinV2_B_384', 'DPT_SwinV2_L_384']:
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        rgb = torch.stack([x['original_rgb'] for x in sample])
        rgb = F.interpolate(rgb, [384, 384])
        return transform(rgb / 255)

def create_bases(disp):
    B, C, H, W = disp.shape
    assert C == 1
    cx = 0.5
    cy = 0.5

    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
    xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
    u, v = torch.meshgrid(xs, ys, indexing='xy')
    u = u - cx
    v = v - cy
    u = u.unsqueeze(0).unsqueeze(0)
    v = v.unsqueeze(0).unsqueeze(0)
    u = u.repeat(B, 1, 1, 1).cuda()
    v = v.repeat(B, 1, 1, 1).cuda()

    aspect_ratio = W / H

    Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
    Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
    Tz = torch.cat([u, v], dim=1)

    Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
    Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
    Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)
    
    Tx = 2 * disp * Tx
    Ty = 2 * disp * Ty
    Tz = 2 * disp * Tz

    R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
    R2x = torch.cat([u * v, v * v], dim=1)
    R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
    R2y = torch.cat([-u * u, -u * v], dim=1)
    Rz =  torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

    R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1,2,3), keepdim=True)
    R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1,2,3), keepdim=True)
    R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1,2,3), keepdim=True)
    R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1,2,3), keepdim=True)
    Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)
    
    M = torch.stack([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=0) # 8xBx2xHxW
    return M


def create_bases_6(disp):
    B, C, H, W = disp.shape
    assert C == 1
    cx = 0.5
    cy = 0.5
    fx = 0.58
    fy = 1.92

    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
    xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
    u, v = torch.meshgrid(xs, ys, indexing='xy')
    u = u - cx
    v = v - cy
    u = u.unsqueeze(0).unsqueeze(0)
    v = v.unsqueeze(0).unsqueeze(0)
    u = u.repeat(B, 1, 1, 1).cuda()
    v = v.repeat(B, 1, 1, 1).cuda()

    # aspect_ratio = W / H
    aspect_ratio = fy / fx

    Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
    Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
    Tz = torch.cat([u, v], dim=1)

    Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
    Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
    Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)
    
    Tx = 2 * disp * Tx
    Ty = 2 * disp * Ty
    Tz = 2 * disp * Tz

    Rx = torch.cat([(u * v) / fy, fy +  (v * v) / fy], dim=1)
    Ry = torch.cat([fx + (u * u) / fx, (u * v) / fx], dim=1)
    Rz =  torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

    Rx = Rx / torch.linalg.vector_norm(Rx, dim=(1,2,3), keepdim=True)
    Ry = Ry / torch.linalg.vector_norm(Ry, dim=(1,2,3), keepdim=True)
    Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)
    
    M = torch.stack([Tx, Ty, Tz, Rx, Ry, Rz], dim=0) # 6xBx2xHxW
    return M


def project_flow_to_bases(bases, flow):
    # flow:  Bx2xHxW
    # bases: Bx2xHxWxN
    bases = bases + (torch.randn(bases.shape, device=bases.device) * 0.00001)
    bases = bases.permute(0, 4, 2, 3, 1).flatten(2).permute(0, 2, 1)
    try:
        U, S, Vh = torch.linalg.svd(bases, full_matrices=False)
        epsilon = 1e-5
        mask = S >= epsilon
        mask = mask.unsqueeze(1).repeat(1, U.shape[1], 1)
        Us = U * mask

        projected_flow = Us @ (Us.transpose(1, 2) @ flow.permute(0, 2, 3, 1).flatten(1).unsqueeze(2))
        return projected_flow.reshape(flow.permute(0, 2, 3, 1).shape).permute(0, 3, 1, 2)
    except:
        return None

def normalise_disparity(disp):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(disp.max().cpu().data)
    mi = float(disp.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (disp - mi) / d