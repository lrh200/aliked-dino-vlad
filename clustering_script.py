from __future__ import print_function
import os
import sys
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==================== Configuration ====================
seed = 42
num_clusters = 64
max_keypoints = 1024
num_samples = 10000  # Number of features to sample for clustering

# Paths
DatasetDir = "/home/member/chxm/lrh/UAVPairs/uavpairs"
CheckpointsDir = "/home/member/chxm/lrh/UAVPairs/checkpoints"
DatasetPath = os.path.join(DatasetDir, "trainset/images")
hdf5Path = os.path.join(CheckpointsDir, "centroids/ALIKED_DINO_VLAD_64_desc_cen.hdf5")

NewWidth = 480
NewHeight = 320


# ==================== Import Modules ====================
# Import from aliked_modules.py
try:
    from aliked_modules import get_patches, simple_nms, InputPadder, DKD, SDDH
except ImportError:
    print("Error: aliked_modules.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


# ==================== ALIKED Feature Extractor ====================
class ALIKED(nn.Module):
    def __init__(self, model_name="aliked-n16", max_num_keypoints=1024, detection_threshold=0.2):
        super().__init__()
        
        cfgs = {
            "aliked-n16": {"c1": 16, "c2": 32, "c3": 64, "c4": 128, "dim": 128, "K": 3, "M": 16},
        }
        
        c1, c2, c3, c4, dim, K, M = [v for _, v in cfgs[model_name].items()]
        
        self.gate = nn.SELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Encoder blocks
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
        print(f"Loading DINOv2 model: {weights}")
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


# ==================== Feature Extraction Model ====================
class FeatureExtractor(nn.Module):
    """Extract fused features without VLAD pooling"""
    def __init__(self, max_keypoints=1024):
        super(FeatureExtractor, self).__init__()
        
        self.aliked = ALIKED(model_name="aliked-n16", max_num_keypoints=max_keypoints, detection_threshold=0.2)
        self.dino = DinoV2(weights="dinov2_vits14", allow_resize=True)
        self.fusion = PointPatchFusion(point_dim=128, patch_dim=384, output_dim=256)
        
    def forward(self, x):
        B = x.size(0)
        
        # Extract ALIKED features
        aliked_out = self.aliked({"image": x})
        keypoints_batch = aliked_out["keypoints"]
        descriptors_batch = aliked_out["descriptors"]
        
        # Extract DINOv2 features
        dino_out = self.dino({"image": x})
        dino_features = dino_out["features"]
        
        all_fused_features = []
        
        for i in range(B):
            keypoints = keypoints_batch[i]
            point_desc = descriptors_batch[i]
            
            # Sample DINOv2 features at keypoint locations
            kp_batch = keypoints.unsqueeze(0)
            dino_feat = dino_features[i:i+1]
            patch_desc = self.dino.sample_features(kp_batch, dino_feat)[0]
            
            # Fuse features
            fused_features = self.fusion(point_desc, patch_desc)
            all_fused_features.append(fused_features)
        
        return all_fused_features


def get_image_list(dataset_path, num_images=1000):
    """Get list of training images"""
    image_list = []
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_list.append(os.path.join(root, file))
    
    print(f"Found {len(image_list)} total images")
    
    # Sample random images if there are too many
    if len(image_list) > num_images:
        random.shuffle(image_list)
        image_list = image_list[:num_images]
        print(f"Sampled {num_images} images for clustering")
    
    return image_list


def extract_features(model, image_list, device, batch_size=4):
    """Extract fused features from images"""
    model.eval()
    
    all_features = []
    
    transform = transforms.Compose([
        transforms.Resize((NewHeight, NewWidth)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n===> Extracting features from {len(image_list)} images...")
    with torch.no_grad():
        for i in range(0, len(image_list), batch_size):
            batch_images = []
            batch_end = min(i + batch_size, len(image_list))
            
            if (i // batch_size) % 10 == 0:
                print(f"Processing batch {i//batch_size + 1}/{(len(image_list) + batch_size - 1)//batch_size}")
            
            for img_path in image_list[i:batch_end]:
                try:
                    img = Image.open(img_path)
                    if len(img.split()) != 3:
                        img = img.convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Extract fused features
            try:
                fused_features_batch = model(batch_tensor)
                
                # Collect all features
                for fused_features in fused_features_batch:
                    all_features.append(fused_features.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return all_features


def perform_clustering(features_list, num_clusters, num_samples):
    """Perform k-means clustering on sampled features"""
    print(f"\n===> Collecting features for clustering...")
    
    # Flatten all features
    all_features = []
    for feat in features_list:
        all_features.append(feat)
    
    all_features = np.vstack(all_features)
    print(f"Total features collected: {all_features.shape}")
    
    # Sample features if there are too many
    if all_features.shape[0] > num_samples:
        print(f"Sampling {num_samples} features from {all_features.shape[0]}")
        indices = np.random.choice(all_features.shape[0], num_samples, replace=False)
        sampled_features = all_features[indices]
    else:
        sampled_features = all_features
        print(f"Using all {all_features.shape[0]} features for clustering")
    
    print(f"Sampled features shape: {sampled_features.shape}")
    
    # Perform k-means clustering
    print(f"\n===> Performing k-means clustering with {num_clusters} clusters...")
    print("This may take a few minutes...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10, verbose=1, max_iter=300)
    kmeans.fit(sampled_features)
    
    centroids = kmeans.cluster_centers_
    print(f"Centroids shape: {centroids.shape}")
    print(f"Clustering inertia: {kmeans.inertia_:.2f}")
    
    return centroids, sampled_features


def save_clusters(centroids, descriptors, output_path):
    """Save cluster centers and descriptors to HDF5 file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as h5:
        h5.create_dataset('centroids', data=centroids)
        h5.create_dataset('descriptors', data=descriptors)
    
    print(f"\n===> Cluster centers saved to: {output_path}")


if __name__ == '__main__':
    time1 = time.time()
    
    print("=" * 80)
    print("ALIKED+DINO+VLAD Clustering Script")
    print("=" * 80)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build feature extraction model
    print("\n===> Building feature extraction model...")
    model = FeatureExtractor(max_keypoints=max_keypoints)
    model = model.to(device)
    model.eval()
    print("Model built successfully!")
    
    # Get training images
    print("\n===> Collecting training images...")
    image_list = get_image_list(DatasetPath, num_images=1000)
    
    if len(image_list) == 0:
        print("Error: No images found in dataset path!")
        sys.exit(1)
    
    # Extract features
    features_list = extract_features(model, image_list, device, batch_size=4)
    
    if len(features_list) == 0:
        print("Error: No features extracted!")
        sys.exit(1)
    
    print(f"\nExtracted features from {len(features_list)} images")
    
    # Perform clustering
    try:
        centroids, descriptors = perform_clustering(features_list, num_clusters, num_samples)
    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save cluster centers
    save_clusters(centroids, descriptors, hdf5Path)
    
    time2 = time.time()
    print(f"\n===> Total time: {time2-time1:.2f} seconds")
    print("=" * 80)
    print("Clustering completed successfully!")
    print(f"Cluster file: {hdf5Path}")
    print("You can now run the training script.")
    print("=" * 80)
