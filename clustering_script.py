from __future__ import print_function
import argparse
import math
import os
import sys
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import h5py
import faiss
from tensorboardX import SummaryWriter
import numpy as np
from numpy import mean
from collections import defaultdict
import time
from scipy.spatial import cKDTree

import RankedListdataset

# ==================== Configuration ====================
threads = 12
seed = 42
batchSize = 5
cacheBatchSize = 20
cacheRefreshRate = 0
margin = 0.1
alpha = 1.35
nGPU = 1
LearnRateStep = 5
LearnRateGamma = 0.5
momentum = 0.9
nEpochs = 20
StartEpoch = 0
evalEvery = 1
patience = 0
NewWidth = 480
NewHeight = 320
optimtype = "adam"
LearnRate = 1e-5
weightDecay = 5e-4

num_clusters = 64
max_keypoints = 1024
losstype = "RankedList"

# Paths
DatasetDir = "/home/member/chxm/lrh/UAVPairs/uavpairs"
ProjectDir = "/home/member/chxm/lrh/UAVPairs"
CheckpointsDir = "/home/member/chxm/lrh/UAVPairs/checkpoints"
DatasetPath = os.path.join(DatasetDir, "trainset/images")
TestDatasetPath = os.path.join(DatasetDir, "testset/images/cug/images")
TrainMatPath = os.path.join(DatasetDir, "trainset/BatchedNontrivialSample_train2.mat")
TestMatPath = os.path.join(DatasetDir, "trainset/test.mat")
hdf5Path = os.path.join(CheckpointsDir, "centroids/ALIKED_DINO_VLAD_64_desc_cen.hdf5")
runsPath = os.path.join(CheckpointsDir, "runs/")
gt_test_file = os.path.join(DatasetDir, "trainset/true_pair_100.txt")


# ==================== ALIKED Feature Extractor ====================
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
        
        from .aliked_modules import SDDH, DKD
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=False, mask=False)
        self.dkd = DKD(radius=2, top_k=max_num_keypoints if detection_threshold <= 0 else -1,
                       scores_th=detection_threshold, n_limit=max_num_keypoints if max_num_keypoints > 0 else 20000)

    def extract_dense_map(self, image):
        from .aliked_modules import InputPadder
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
        point_proj = self.point_proj(point_desc)
        patch_proj = self.patch_proj(patch_desc)
        
        concat_features = torch.cat([point_proj, patch_proj], dim=-1)
        fused_features = F.relu(self.fusion(concat_features))
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features


# ==================== VLAD Pooling ====================
class VLADPooling(nn.Module):
    """VLAD pooling for global descriptor aggregation"""
    def __init__(self, num_clusters=64, dim=256):
        super(VLADPooling, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        
        # Learnable cluster centers
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        
        # Soft assignment layer
        self.conv = nn.Linear(dim, num_clusters)
        
    def init_params(self, clsts, traindescs=None):
        """Initialize cluster centers"""
        self.centroids.data = torch.from_numpy(clsts).float()
        
        # Initialize soft assignment weights
        clsts_normalized = clsts / (np.linalg.norm(clsts, axis=1, keepdims=True) + 1e-8)
        self.conv.weight.data = torch.from_numpy(clsts_normalized).float()
        self.conv.bias.data.zero_()
        
    def forward(self, x):
        """
        Args:
            x: (N, dim) - local features
        Returns:
            vlad: (1, num_clusters * dim) - VLAD descriptor
        """
        N, D = x.shape
        
        # Normalize input
        x = F.normalize(x, p=2, dim=1)
        
        # Soft assignment
        soft_assign = self.conv(x)  # (N, num_clusters)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # Calculate residuals
        vlad = torch.zeros(self.num_clusters, D, device=x.device, dtype=x.dtype)
        
        for k in range(self.num_clusters):
            residual = x - self.centroids[k:k+1]
            weighted_residual = residual * soft_assign[:, k:k+1]
            vlad[k] = weighted_residual.sum(dim=0)
        
        # Intra-normalization
        vlad = F.normalize(vlad, p=2, dim=1)
        
        # Flatten and L2 normalize
        vlad = vlad.view(-1)
        vlad = F.normalize(vlad, p=2, dim=0)
        
        return vlad.unsqueeze(0)


# ==================== Complete Model ====================
class ALIKEDDinoVLAD(nn.Module):
    """Complete (ALIKED + DINO) + VLAD model"""
    def __init__(self, num_clusters=64, max_keypoints=1024):
        super(ALIKEDDinoVLAD, self).__init__()
        
        self.aliked = ALIKED(model_name="aliked-n16", max_num_keypoints=max_keypoints, detection_threshold=0.2)
        self.dino = DinoV2(weights="dinov2_vits14", allow_resize=True)
        self.fusion = PointPatchFusion(point_dim=128, patch_dim=384, output_dim=256)
        self.vlad = VLADPooling(num_clusters=num_clusters, dim=256)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - batch of images
        Returns:
            global_descriptors: (B, num_clusters * dim) - batch of VLAD descriptors
        """
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
        
        # Stack all descriptors
        global_descriptors = torch.cat(global_descriptors, dim=0)
        
        return global_descriptors


# ==================== Training Functions ====================
def TrainOneEpochTriplet(epoch):
    TrainData.GetDataType = 'RankedList'
    epoch_loss = 0
    startIter = 1
    
    if cacheRefreshRate > 0:
        subsetN = ceil(len(TrainData) / cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(TrainData)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(TrainData))]
    
    nBatches = (len(TrainData) + batchSize - 1) // batchSize
    print("number of batches: ", nBatches)
    print("Divide TrainData into", subsetN, "groups")

    for subIter in range(subsetN):
        print("Currently the number ", str(subIter + 1), "group of TrainData")

        model.eval()
        SubData = Subset(dataset=TrainData, indices=subsetIdx[subIter])
        SubQueryDataLoader = DataLoader(
            dataset=SubData,
            num_workers=threads,
            batch_size=batchSize,
            shuffle=False,
            collate_fn=RankedListdataset.collate_fn,
            pin_memory=True
        )
        
        model.train()
        
        for iteration, (query, positives, posCounts, index) in enumerate(SubQueryDataLoader, startIter):
            if query is None:
                continue
            
            B, C, H, W = query.shape
            nPos = torch.sum(posCounts)
            
            input = torch.cat([query, positives])
            input = input.to(device)
            
            # Forward pass through the model
            vlad_encoding = model(input)
            
            vladQ, vladP = torch.split(vlad_encoding, [B, nPos])
            
            optimizer.zero_grad()
            
            loss = torch.tensor(0.0).to(device)
            count = torch.tensor(0.0).to(device)
            
            for i, posCount in enumerate(posCounts):
                for j in range(posCount):
                    posIx = (torch.sum(posCounts[:i]) + j).item()
                    for k in range(B):
                        if k != i:
                            for nj in range(posCounts[k]):
                                negIx = (torch.sum(posCounts[:k]) + nj).item()
                                loss_item = criterion(vladQ[i], vladP[posIx], vladP[negIx])
                                if loss_item > 0:
                                    loss += loss_item
                                    count += 1
            
            loss /= (nPos * (nPos - int(nPos/B))).float().to(device)
            loss /= (count + 1e-6).float().to(device)
            
            if loss <= 0:
                continue
                
            loss.backward()
            optimizer.step()
            
            del input, vlad_encoding, vladQ, vladP
            del query, positives
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            if iteration % 100 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                    epoch, iteration, nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                ((epoch - 1) * nBatches) + iteration)

        startIter += len(SubQueryDataLoader)
        del SubQueryDataLoader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        TrainData.GetDataType = 'None'

    avg_loss = epoch_loss / nBatches
    print("===> Epoch {} Complete!  Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    return avg_loss


def TrainOneEpoch(epoch):
    TrainData.GetDataType = 'RankedList'
    epoch_loss = 0
    epoch_loss_p = 0
    epoch_loss_n = 0
    startIter = 1
    
    if cacheRefreshRate > 0:
        subsetN = ceil(len(TrainData) / cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(TrainData)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(TrainData))]
    
    nBatches = (len(TrainData) + batchSize - 1) // batchSize
    print("number of batches: ", nBatches)
    print("Divide TrainData into", subsetN, "groups")

    for subIter in range(subsetN):
        print("Currently the number ", str(subIter + 1), "group of TrainData")
        model.eval()

        SubData = Subset(dataset=TrainData, indices=subsetIdx[subIter])
        SubQueryDataLoader = DataLoader(
            dataset=SubData,
            num_workers=threads,
            batch_size=batchSize,
            shuffle=False,
            collate_fn=RankedListdataset.collate_fn,
            pin_memory=True
        )
        
        model.train()
        
        for iteration, (query, positives, posCounts, index) in enumerate(SubQueryDataLoader, startIter):
            if query is None:
                continue
            
            B, C, H, W = query.shape
            nPos = torch.sum(posCounts)
            
            input = torch.cat([query, positives])
            input = input.to(device)
            
            # Forward pass
            vlad_encoding = model(input)
            
            vladQ, vladP = torch.split(vlad_encoding, [B, nPos])
            
            optimizer.zero_grad()
            
            loss_p = torch.tensor(0.0).to(device)
            loss_n = torch.tensor(0.0).to(device)
            count_n = torch.tensor(0.0).to(device)
            count_p = torch.tensor(0.0).to(device)
            
            for i, posCount in enumerate(posCounts):
                for j in range(posCount):
                    posIx = (torch.sum(posCounts[:i]) + j).item()
                    dist_ap = pdist(vladQ[i], vladP[posIx])
                    dist_aa = torch.zeros_like(dist_ap)
                    y = torch.ones_like(dist_ap)
                    loss_p_item = criterionP(dist_aa, dist_ap, y)
                    if loss_p_item > 0:
                        loss_p += loss_p_item
                        count_p += 1

                    for k in range(B):
                        if k != i:
                            dist_an = pdist(vladQ[k], vladP[posIx])
                            dist_aa = torch.zeros_like(dist_an)
                            y = torch.ones_like(dist_an)
                            loss_n_item = criterionN(dist_an, dist_aa, y)
                            if loss_n_item > 0:
                                loss_n += loss_n_item
                                count_n += 1

            loss_p /= (count_p + 1e-6).float().to(device)
            loss_n /= (count_n + 1e-6).float().to(device)
            loss = loss_p + loss_n

            if loss <= 0:
                continue
            loss.backward()
            optimizer.step()

            del input, vlad_encoding, vladQ, vladP
            del query, positives

            batch_loss = loss.item()
            batch_loss_n = loss_n.item()
            batch_loss_p = loss_p.item()

            epoch_loss += batch_loss
            epoch_loss_n += batch_loss_n
            epoch_loss_p += batch_loss_p

            if iteration % 100 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}, Loss_p: {:.4f}, Loss_n: {:.4f}".format(
                    epoch, iteration, nBatches, batch_loss, batch_loss_p, batch_loss_n), flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                ((epoch - 1) * nBatches) + iteration)

        startIter += len(SubQueryDataLoader)
        del SubQueryDataLoader, loss_p, loss_n, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        TrainData.GetDataType = 'None'

    avg_loss = epoch_loss / nBatches
    avg_loss_p = epoch_loss_p / nBatches
    avg_loss_n = epoch_loss_n / nBatches
    print("===> Epoch {} Complete!  Avg. Loss: {:.4f}, Loss_p: {:.4f}, Loss_n: {:.4f}".format(
        epoch, avg_loss, avg_loss_p, avg_loss_n), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    return avg_loss


def Validate_campus(TestData, epoch=0, write_tboard=False):
    time1 = time.time()
    model.eval()

    QueryFeat = torch.zeros(0).to(device)

    TestData.GetDataType = 'TestQuery'
    Databaseloader = DataLoader(
        dataset=TestData,
        num_workers=8,
        batch_size=5,
        shuffle=False,
        pin_memory=True
    )
    
    with torch.no_grad():
        for iteration, (QueryImg, Index) in enumerate(Databaseloader, 1):
            QueryImg = QueryImg.to(device)
            QueryFeat_batch = model(QueryImg)
            QueryFeat = torch.cat((QueryFeat, QueryFeat_batch), 0)

    QueryFeat = QueryFeat.cpu().detach().numpy()
    print(QueryFeat.shape)
    
    test_dict = defaultdict(list)
    gt_dict = defaultdict(list)

    netvlad_index = faiss.IndexFlatL2(faiss_dim)
    netvlad_index.add(QueryFeat)

    num_test = 0
    D, I = netvlad_index.search(QueryFeat, 31)
    del QueryFeat
    
    for index, values in enumerate(I):
        im1 = TestData.TestDatabase[index].replace(TestDatasetPath + "\\", '')
        for value in values[1:]:
            im2 = TestData.TestDatabase[value].replace(TestDatasetPath + "\\", '')
            test_dict[im1].append(im2)
            num_test = num_test + 1

    with open(gt_test_file, 'r') as lines:
        for line in lines:
            line = line.strip('\n')
            strs = line.split(" ")
            gt_dict[strs[0]].append(strs[1])
            gt_dict[strs[1]].append(strs[0])

    global_num = 0
    for key in test_dict:
        set_c = set(test_dict[key]) & set(gt_dict[key])
        list_c = list(set_c)
        local_num = len(list_c)
        global_num = global_num + local_num
    mAP = global_num / (num_test * 1.0)

    print("-------------------------------")
    print("===> global_num: {}".format(global_num))
    print("====> mAP: {:.5f}".format(mAP))
    time2 = time.time()
    print(time2-time1)
    print("-------------------------------")
    torch.cuda.empty_cache()
    return mAP


def save_checkpoint(state, is_best, filename):
    model_out_path = join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))


# ==================== Main Training Script ====================
if __name__ == '__main__':
    print("Model: ALIKED+DINO+VLAD")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading training dataset(s)...')
    TrainData = RankedListdataset.Dataset(TrainMatPath, DatasetPath, True)
    TrainDataLoader = DataLoader(
        dataset=TrainData,
        num_workers=threads,
        batch_size=cacheBatchSize,
        shuffle=False,
        pin_memory=True
    )

    TestData = RankedListdataset.Dataset(TestMatPath, TestDatasetPath, False)
    TestDataLoader = DataLoader(
        dataset=TestData,
        num_workers=threads,
        batch_size=cacheBatchSize,
        shuffle=False,
        pin_memory=True
    )
    
    print('Number of original triples:', len(TrainData.Query))
    print('Number of triples after filtering:', len(TrainData))
    print('Number of database images:', len(TrainData.Database))

    print('===> Building ALIKED+DINO+VLAD model...')
    
    model = ALIKEDDinoVLAD(num_clusters=num_clusters, max_keypoints=max_keypoints)
    
    # Load VLAD cluster centers if available
    if os.path.exists(hdf5Path):
        print(f"Loading cluster centers from {hdf5Path}")
        with h5py.File(hdf5Path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            model.vlad.init_params(clsts)
            del clsts
    else:
        print(f"Warning: Cluster centers not found at {hdf5Path}")
        print("Please run clustering first!")
        # You can still train, but clustering should be done for better results

    model = model.to(device)
    print("Network Structure:")
    print(model)
    
    # Calculate output dimension
    faiss_dim = num_clusters * 256  # VLAD output dimension

    # Define optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimtype == 'sgd':
        optimizer = optim.SGD(parameters, LearnRate, momentum=momentum, weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LearnRateStep, gamma=LearnRateGamma)
    elif optimtype == 'adam':
        optimizer = optim.Adam(parameters, LearnRate, weight_decay=weightDecay)
        exp_decay = math.exp(-0.1)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # Define loss functions
    if losstype == "Triplet":
        criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)
    else:
        criterionP = nn.MarginRankingLoss(margin=margin-alpha, reduction='none').to(device)
        criterionN = nn.MarginRankingLoss(margin=alpha, reduction='none').to(device)
        pdist = nn.PairwiseDistance(p=2)

    print('===> Training model...')

    OutputFile = "ALIKED_DINO_VLAD.txt"
    with open(OutputFile, 'a') as Output:
        Output.write("ALIKED_DINO_VLAD\n")
    
    writer = SummaryWriter(
        log_dir=join(runsPath, datetime.now().strftime('%b%d_%H-%M-%S') + '_ALIKED_DINO_VLAD')
    )
    logdir = writer.file_writer.get_logdir()
    savePath = join(logdir, "checkpoints")
    
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    makedirs(savePath)

    BestmAP = 0

    # Initial validation
    mAP = Validate_campus(TestData, 0, write_tboard=True)
    with open(OutputFile, 'a') as Output:
        Output.write(str(0) + "\t" + str(alpha-margin) + "\t" + str(mAP) + "\n")
    
    CheckPointFile = "CheckPoint_ALIKED_DINO_VLAD_" + str(0) + ".pth.tar"
    save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'mAP': mAP,
        'best_score': mAP,
        'optimizer': optimizer.state_dict(),
        'parallel': False,
    }, False, CheckPointFile)

    # Training loop
    for epoch in range(StartEpoch + 1, nEpochs + 1):
        if losstype == "Triplet":
            AveLoss = TrainOneEpochTriplet(epoch)
        else:
            AveLoss = TrainOneEpoch(epoch)
        
        scheduler.step(epoch)
        mAP = Validate_campus(TestData, epoch, write_tboard=True)
        
        if mAP > BestmAP:
            BestmAP = mAP
            IsBestFlag = True
        else:
            IsBestFlag = False

        with open(OutputFile, 'a') as Output:
            Output.write(str(epoch) + "\t" + str(AveLoss) + "\t" + str(mAP) + "\n")
        
        CheckPointFile = "CheckPoint_ALIKED_DINO_VLAD_" + str(epoch) + ".pth.tar"
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'mAP': mAP,
            'best_score': BestmAP,
            'optimizer': optimizer.state_dict(),
            'parallel': False,
        }, IsBestFlag, CheckPointFile)

    print("Training completed!")
    writer.close()
