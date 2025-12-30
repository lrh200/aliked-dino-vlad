from __future__ import print_function

import glob
import sys
import random
from os.path import join, isfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
import time
from PIL import ImageFile
from PIL import Image
import torchvision.transforms as transforms
import sqlite3 as sqlite
from scipy.spatial import cKDTree
from omegaconf import OmegaConf

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==================== ALIKED Components ====================
def get_patches(tensor: torch.Tensor, required_corners: torch.Tensor, ps: int):
    c, h, w = tensor.shape
    corner = (required_corners - ps / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    return sampled.permute(2, 3, 0, 1)


def simple_nms(scores: torch.Tensor, nms_radius: int):
    zeros = torch.zeros_like(scores)
    max_mask = scores == F.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )

    for _ in range(2):
        supp_mask = (
            F.max_pool2d(
                max_mask.float(),
                kernel_size=nms_radius * 2 + 1,
                stride=1,
                padding=nms_radius,
            )
            > 0
        )
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == F.max_pool2d(
            supp_scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class InputPadder:
    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class DKD(nn.Module):
    def __init__(self, radius: int = 2, top_k: int = 0, scores_th: float = 0.2, n_limit: int = 20000):
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
        self.hw_grid = torch.stack(torch.meshgrid([x, x], **kw)).view(2, -1).t()[:, [1, 0]]

    def forward(self, scores_map: torch.Tensor, sub_pixel: bool = True, image_size=None):
        b, c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        nms_scores = simple_nms(scores_nograd, self.radius)

        nms_scores[:, :, : self.radius, :] = 0
        nms_scores[:, :, :, : self.radius] = 0
        if image_size is not None:
            for i in range(scores_map.shape[0]):
                w_i, h_i = image_size[i].long()
                nms_scores[i, :, h_i.item() - self.radius :, :] = 0
                nms_scores[i, :, :, w_i.item() - self.radius :] = 0
        else:
            nms_scores[:, :, -self.radius :, :] = 0
            nms_scores[:, :, :, -self.radius :] = 0

        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(b, -1), self.top_k)
            indices_keypoints = [topk.indices[i] for i in range(b)]
        else:
            if self.scores_th > 0:
                masks = nms_scores > self.scores_th
                if masks.sum() == 0:
                    th = scores_nograd.reshape(b, -1).mean(dim=1)
                    masks = nms_scores > th.reshape(b, 1, 1, 1)
            else:
                th = scores_nograd.reshape(b, -1).mean(dim=1)
                masks = nms_scores > th.reshape(b, 1, 1, 1)
            masks = masks.reshape(b, -1)

            indices_keypoints = []
            scores_view = scores_nograd.reshape(b, -1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero()[:, 0]
                if len(indices) > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    sel_idx = sort_idx[: self.n_limit]
                    indices = indices[sel_idx]
                indices_keypoints.append(indices)

        wh = torch.tensor([w - 1, h - 1], device=scores_nograd.device)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        
        if sub_pixel:
            patches = self.unfold(scores_map)
            self.hw_grid = self.hw_grid.to(scores_map)
            for b_idx in range(b):
                patch = patches[b_idx].t()
                indices_kpt = indices_keypoints[b_idx]
                patch_scores = patch[indices_kpt]
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, torch.div(indices_kpt, w, rounding_mode="trunc")], dim=1
                )

                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = ((patch_scores - max_v) / self.temperature).exp()

                xy_residual = x_exp @ self.hw_grid / x_exp.sum(dim=1)[:, None]

                hw_grid_dist2 = (
                    torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius, dim=-1) ** 2
                )
                scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = keypoints_xy / wh * 2 - 1

                kptscore = F.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0, 0, :]

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[b_idx]
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, torch.div(indices_kpt, w, rounding_mode="trunc")], dim=1
                )
                keypoints_xy = keypoints_xy_nms / wh * 2 - 1
                kptscore = F.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0, 0, :]
                keypoints.append(keypoints_xy)
                scoredispersitys.append(kptscore)
                kptscores.append(kptscore)

        return keypoints, scoredispersitys, kptscores


class SDDH(nn.Module):
    def __init__(self, dims: int, kernel_size: int = 3, n_pos: int = 8, gate=nn.ReLU(), conv2D=False, mask=False):
        super(SDDH, self).__init__()
        self.kernel_size = kernel_size
        self.n_pos = n_pos
        self.conv2D = conv2D
        self.mask = mask
        self.get_patches_func = get_patches

        self.channel_num = 3 * n_pos if mask else 2 * n_pos
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dims, self.channel_num, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            gate,
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.sf_conv = nn.Conv2d(dims, dims, kernel_size=1, stride=1, padding=0, bias=False)

        if not conv2D:
            agg_weights = torch.nn.Parameter(torch.rand(n_pos, dims, dims))
            self.register_parameter("agg_weights", agg_weights)
        else:
            self.convM = nn.Conv2d(dims * n_pos, dims, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, keypoints):
        b, c, h, w = x.shape
        wh = torch.tensor([[w - 1, h - 1]], device=x.device)
        max_offset = max(h, w) / 4.0

        descriptors = []
        for ib in range(b):
            xi, kptsi = x[ib], keypoints[ib]
            kptsi_wh = (kptsi / 2 + 0.5) * wh
            N_kpts = len(kptsi)

            if self.kernel_size > 1:
                patch = self.get_patches_func(xi, kptsi_wh.long(), self.kernel_size)
            else:
                kptsi_wh_long = kptsi_wh.long()
                patch = xi[:, kptsi_wh_long[:, 1], kptsi_wh_long[:, 0]].permute(1, 0).reshape(N_kpts, c, 1, 1)

            offset = self.offset_conv(patch).clamp(-max_offset, max_offset)
            if self.mask:
                offset = offset[:, :, 0, 0].view(N_kpts, 3, self.n_pos).permute(0, 2, 1)
                offset = offset[:, :, :-1]
                mask_weight = torch.sigmoid(offset[:, :, -1])
            else:
                offset = offset[:, :, 0, 0].view(N_kpts, 2, self.n_pos).permute(0, 2, 1)

            pos = kptsi_wh.unsqueeze(1) + offset
            pos = 2.0 * pos / wh[None] - 1
            pos = pos.reshape(1, N_kpts * self.n_pos, 1, 2)

            features = F.grid_sample(xi.unsqueeze(0), pos, mode="bilinear", align_corners=True)
            features = features.reshape(c, N_kpts, self.n_pos, 1).permute(1, 0, 2, 3)
            if self.mask:
                features = torch.einsum("ncpo,np->ncpo", features, mask_weight)

            features = torch.selu_(self.sf_conv(features)).squeeze(-1)
            if not self.conv2D:
                descs = torch.einsum("ncp,pcd->nd", features, self.agg_weights)
            else:
                features = features.reshape(N_kpts, -1)[:, :, None, None]
                descs = self.convM(features).squeeze()

            descs = F.normalize(descs, p=2.0, dim=1)
            descriptors.append(descs)

        return descriptors


class ALIKED(nn.Module):
    def __init__(self, model_name="aliked-n16", max_num_keypoints=1024, detection_threshold=0.2):
        super().__init__()
        from torchvision.models import resnet
        
        cfgs = {
            "aliked-n16": {"c1": 16, "c2": 32, "c3": 64, "c4": 128, "dim": 128, "K": 3, "M": 16},
        }
        
        c1, c2, c3, c4, dim, K, M = [v for _, v in cfgs[model_name].items()]
        
        self.gate = nn.SELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Encoder blocks (simplified)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), self.gate,
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), self.gate
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), self.gate,
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), self.gate
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), self.gate,
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), self.gate
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), self.gate,
            nn.Conv2d(c4, c4, 3, padding=1), nn.BatchNorm2d(c4), self.gate
        )
        
        self.conv1 = nn.Conv2d(c1, dim // 4, 1)
        self.conv2 = nn.Conv2d(c2, dim // 4, 1)
        self.conv3 = nn.Conv2d(c3, dim // 4, 1)
        self.conv4 = nn.Conv2d(c4, dim // 4, 1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)
        
        self.score_head = nn.Sequential(
            nn.Conv2d(dim, 8, 1), self.gate,
            nn.Conv2d(8, 4, 3, padding=1), self.gate,
            nn.Conv2d(4, 4, 3, padding=1), self.gate,
            nn.Conv2d(4, 1, 3, padding=1),
        )
        
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=False, mask=False)
        self.dkd = DKD(radius=2, top_k=max_num_keypoints if detection_threshold <= 0 else -1,
                       scores_th=detection_threshold, n_limit=max_num_keypoints if max_num_keypoints > 0 else 20000)

    def extract_dense_map(self, image):
        div_by = 32
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        x1 = self.block1(image)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        x3 = self.pool4(x2)
        x3 = self.block3(x3)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)

        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))
        
        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        
        score_map = torch.sigmoid(self.score_head(x1234))
        feature_map = F.normalize(x1234, p=2, dim=1)

        feature_map = padder.unpad(feature_map)
        score_map = padder.unpad(score_map)

        return feature_map, score_map

    def forward(self, data):
        image = data["image"]
        feature_map, score_map = self.extract_dense_map(image)
        keypoints, kptscores, scoredispersitys = self.dkd(score_map, image_size=data.get("image_size"))
        descriptors = self.desc_head(feature_map, keypoints)

        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        
        return {
            "keypoints": [wh * (kp + 1) / 2.0 for kp in keypoints],
            "descriptors": descriptors,
            "keypoint_scores": kptscores,
        }


# ==================== DINOv2 Feature Extractor ====================
class DinoV2(nn.Module):
    def __init__(self, weights="dinov2_vits14", allow_resize=True):
        super().__init__()
        self.allow_resize = allow_resize
        self.net = torch.hub.load("facebookresearch/dinov2", weights)
        self.vit_size = 14
        
    def forward(self, data):
        img = data["image"]
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        if self.allow_resize:
            img = F.interpolate(img, [int(x // 14 * 14) for x in img.shape[-2:]])

        desc, cls_token = self.net.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]
        
        return {
            "features": desc,
            "global_descriptor": cls_token,
            "descriptors": desc.flatten(-2).transpose(-2, -1),
        }

    def sample_features(self, keypoints, features, s=None, mode="bilinear"):
        if s is None:
            s = self.vit_size
        b, c, h, w = features.shape
        keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1
        features = F.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        features = F.normalize(features.reshape(b, c, -1), p=2, dim=1)
        features = features.permute(0, 2, 1)
        return features


# ==================== Feature Fusion Module ====================
class PointPatchFusion(nn.Module):
    def __init__(self, point_dim=128, patch_dim=384, output_dim=256):
        super(PointPatchFusion, self).__init__()
        
        self.point_proj = nn.Linear(point_dim, output_dim)
        self.patch_proj = nn.Linear(patch_dim, output_dim)
        self.fusion = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, point_desc, patch_desc):
        """
        Args:
            point_desc: (N, 128) - point descriptors
            patch_desc: (N, 384) - patch descriptors (already sampled at keypoint locations)
        Returns:
            fused_features: (N, output_dim)
        """
        point_proj = self.point_proj(point_desc)
        patch_proj = self.patch_proj(patch_desc)
        
        concat_features = torch.cat([point_proj, patch_proj], dim=-1)
        fused_features = F.relu(self.fusion(concat_features))
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features


# ==================== VLAD Pooling ====================
class VLADPooling(nn.Module):
    def __init__(self, num_clusters=64, dim=256):
        super(VLADPooling, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.conv = nn.Linear(dim, num_clusters)
        
    def init_params(self, clsts, traindescs=None):
        self.centroids.data = torch.from_numpy(clsts).float()
        clsts_normalized = clsts / (np.linalg.norm(clsts, axis=1, keepdims=True) + 1e-8)
        self.conv.weight.data = torch.from_numpy(clsts_normalized).float()
        self.conv.bias.data.zero_()
        
    def forward(self, x):
        N, D = x.shape
        x = F.normalize(x, p=2, dim=1)
        
        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        vlad = torch.zeros(self.num_clusters, D, device=x.device, dtype=x.dtype)
        
        for k in range(self.num_clusters):
            residual = x - self.centroids[k:k+1]
            weighted_residual = residual * soft_assign[:, k:k+1]
            vlad[k] = weighted_residual.sum(dim=0)
        
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.view(-1)
        vlad = F.normalize(vlad, p=2, dim=0)
        
        return vlad.unsqueeze(0)


# ==================== Complete Model ====================
class ALIKEDDinoVLAD(nn.Module):
    def __init__(self, num_clusters=64, max_keypoints=1024):
        super(ALIKEDDinoVLAD, self).__init__()
        
        self.aliked = ALIKED(model_name="aliked-n16", max_num_keypoints=max_keypoints, detection_threshold=0.2)
        self.dino = DinoV2(weights="dinov2_vits14", allow_resize=True)
        self.fusion = PointPatchFusion(point_dim=128, patch_dim=384, output_dim=256)
        self.vlad = VLADPooling(num_clusters=num_clusters, dim=256)
        
    def forward(self, x):
        B = x.size(0)
        
        # Extract ALIKED features
        aliked_out = self.aliked({"image": x})
        keypoints_batch = aliked_out["keypoints"]  # List of [N, 2]
        descriptors_batch = aliked_out["descriptors"]  # List of [N, 128]
        
        # Extract DINOv2 features
        dino_out = self.dino({"image": x})
        dino_features = dino_out["features"]  # [B, C, H, W]
        
        global_descriptors = []
        
        for i in range(B):
            keypoints = keypoints_batch[i]  # [N, 2]
            point_desc = descriptors_batch[i]  # [N, 128]
            
            # Sample DINOv2 features at keypoint locations
            kp_batch = keypoints.unsqueeze(0)  # [1, N, 2]
            dino_feat = dino_features[i:i+1]  # [1, C, H, W]
            patch_desc = self.dino.sample_features(kp_batch, dino_feat)[0]  # [N, 384]
            
            # Fuse features
            fused_features = self.fusion(point_desc, patch_desc)
            
            # VLAD pooling
            global_desc = self.vlad(fused_features)
            global_descriptors.append(global_desc)
        
        global_descriptors = torch.cat(global_descriptors, dim=0)
        
        return global_descriptors


# ==================== Utility Functions ====================
def getImageList(db_path, test_dataset_path):
    db_con = sqlite.connect(db_path)
    db_cur = db_con.cursor()
    db_cur.execute("select name from images order by image_id")
    res = db_cur.fetchall()
    db_cur.close()
    db_con.close()
    
    im_list = []
    for name in res:
        path_str = name[0]
        relative_path = path_str.split('/cug/images/')[-1]
        im_name = os.path.join(test_dataset_path, relative_path)
        im_list.append(im_name)
    return im_list


# ==================== Main Script ====================
if __name__ == '__main__':
    time1 = time.time()
    
    # Configuration
    seed = 42
    num_clusters = 64
    max_keypoints = 1024
    
    # Paths
    DatasetDir = "/home/member/chxm/lrh/UAVPairs/uavpairs"
    ProjectDir = "/home/member/chxm/lrh/UAVPairs"
    CheckpointsDir = "/home/member/chxm/lrh/UAVPairs/checkpoints"
    TestDatasetPath = os.path.join(DatasetDir, "testset/images/cug/images")
    
    hdf5Path = os.path.join(CheckpointsDir, "centroids/ALIKED_DINO_VLAD_64_desc_cen.hdf5")
    database_path = os.path.join(DatasetDir, "testset/database/campus_test.db")
    output_path = os.path.join(DatasetDir, "testset/campus_ALIKED_DINO_VLAD.npy")
    
    CheckPointPath = os.path.join(CheckpointsDir, "runs/ALIKED_DINO_VLAD/checkpoints")
    CheckPointFile = join(CheckPointPath, 'model_best.pth.tar')
    
    imageList = getImageList(database_path, TestDatasetPath)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print('===> Building ALIKED+DINO+VLAD model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ALIKEDDinoVLAD(num_clusters=num_clusters, max_keypoints=max_keypoints)
    
    # Load VLAD cluster centers
    if os.path.exists(hdf5Path):
        print(f"Loading cluster centers from {hdf5Path}")
        with h5py.File(hdf5Path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            model.vlad.init_params(clsts)
            del clsts
    else:
        print(f"Warning: Cluster centers not found at {hdf5Path}")
        print("Using random initialization. Run clustering first for better results.")
    
    model = model.to(device)
    print("Model built successfully!")
    
    # Load checkpoint if exists
    if isfile(CheckPointFile):
        print(f"=> loading checkpoint '{CheckPointFile}'")
        checkpoint = torch.load(CheckPointFile, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"=> loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"Warning: No checkpoint found at {CheckPointFile}")
        print("Using randomly initialized model.")
    
    # Extract features
    print('===> Start feature extraction')
    model.eval()
    
    DbImageFeat = []
    
    with torch.no_grad():
        for i, image_name in enumerate(imageList):
            print(f"Processing [{i+1}/{len(imageList)}]: {image_name}")
            
            try:
                # Load image
                img = Image.open(image_name)
                if len(img.split()) != 3:
                    img = img.convert('RGB')
                
                # Preprocess - normalize for DINOv2
                img_tensor = transforms.Compose([
                    transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])(img)
                
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # Extract global descriptor
                descriptor = model(img_tensor)
                
                # Move to CPU
                DbImageFeat.append(descriptor.cpu().numpy())
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Stack and save features
    if len(DbImageFeat) > 0:
        DbImageFeat = np.vstack(DbImageFeat)
        np.save(output_path, DbImageFeat)
        
        print(f"Feature extraction complete!")
        print(f"Feature shape: {DbImageFeat.shape}")
        print(f"Features saved to: {output_path}")
    else:
        print("No features extracted!")
    
    time2 = time.time()
    print(f"Total time: {time2-time1:.2f} seconds")
