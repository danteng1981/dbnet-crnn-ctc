"""
DBNet:  Differentiable Binarization Network for Text Detection

Reference: https://arxiv.org/abs/1911.08947
"""

import torch
import torch.nn as nn
import torch. nn.functional as F


class DifferentiableBinarization(nn.Module):
    """可微分二值化模块"""
    
    def __init__(self, k=50):
        super().__init__()
        self.k = k
    
    def forward(self, prob_map, threshold_map):
        """
        Args:
            prob_map: [B, 1, H, W] 概率图
            threshold_map:  [B, 1, H, W] 阈值图
        
        Returns:
            binary_map: [B, 1, H, W] 近似二值图
        """
        binary_map = 1. 0 / (1.0 + torch. exp(-self.k * (prob_map - threshold_map)))
        return binary_map


class DBHead(nn.Module):
    """DBNet 预测头"""
    
    def __init__(self, in_channels, k=50):
        super().__init__()
        
        # 概率图分支
        self.prob_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        
        # 阈值图分支
        self.thresh_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        
        # 可微分二值化
        self.diff_bin = DifferentiableBinarization(k=k)
    
    def forward(self, x):
        prob_map = self.prob_conv(x)
        thresh_map = self.thresh_conv(x)
        
        if self.training:
            binary_map = self.diff_bin(prob_map, thresh_map)
            return prob_map, thresh_map, binary_map
        else:
            return prob_map, thresh_map


class FPN(nn.Module):
    """特征金字塔网络"""
    
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1)
            for _ in in_channels
        ])
    
    def forward(self, features):
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
        
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]
        
        target_size = outputs[0].shape[2:]
        upsampled = [
            F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
            for output in outputs
        ]
        
        fused = torch.cat(upsampled, dim=1)
        return fused


class DBNet(nn.Module):
    """完整 DBNet 模型"""
    
    def __init__(self, backbone='resnet18', pretrained=True, k=50):
        super().__init__()
        
        if backbone == 'resnet18': 
            from torchvision.models import resnet18
            resnet = resnet18(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet. layer4
            in_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.fpn = FPN(in_channels, 256)
        self.head = DBHead(256, k=k)
    
    def forward(self, x):
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        features = self.fpn([c1, c2, c3, c4])
        return self.head(features)
