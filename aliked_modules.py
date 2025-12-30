"""
ALIKED Support Modules
Contains helper functions and classes for ALIKED feature extraction
Save this as: aliked_modules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_patches(tensor: torch.Tensor, required_corners: torch.Tensor, ps: int):
    """Extract patches from tensor at specified corners"""
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
    """Fast Non-maximum suppression"""
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
    """Pads images such that dimensions are divisible by specified value"""
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
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class DKD(nn.Module):
    """Differentiable Keypoint Detection"""
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

        # Remove border
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

        # Select keypoints
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
    """Scale-space Description with Deformable Handling"""
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
