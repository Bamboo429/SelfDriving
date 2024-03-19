#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:17:32 2024

@author: chuhsuanlin
"""

import torch
import torchvision
import torch.nn as nn
from pyramidpooling import SpatialPyramidPooling
from collections import OrderedDict

class Encoder(torch.nn.Module):
    
    def __init__(self, lidar_channel, map_channel):
        super(Encoder, self).__init__()
        
        self.lidar = LiDAR(lidar_channel)
        self.map = Map(map_channel)
        
        self.bottleneck_block1 = nn.Sequential(
            BottleNeck(in_channel=96, out_channel=128, hidden_channel=32, stride=2),
            BottleNeck(in_channel=128, out_channel=128, hidden_channel=32, stride=1)
            )
        
        self.bottleneck_block2 = nn.Sequential(
            BottleNeck(in_channel=128, out_channel=192, hidden_channel=48, stride=2),
            BottleNeck(in_channel=192, out_channel=192, hidden_channel=48, stride=1),
            BottleNeck(in_channel=192, out_channel=192, hidden_channel=48, stride=1)
            )
        
        self.bottleneck_block3 = nn.Sequential(
            BottleNeck(in_channel=192, out_channel=256, hidden_channel=64, stride=2),
            BottleNeck(in_channel=256, out_channel=256, hidden_channel=64, stride=1),
            BottleNeck(in_channel=256, out_channel=256, hidden_channel=64, stride=1),
            BottleNeck(in_channel=256, out_channel=256, hidden_channel=64, stride=1),
            BottleNeck(in_channel=256, out_channel=256, hidden_channel=64, stride=1),
            BottleNeck(in_channel=256, out_channel=256, hidden_channel=64, stride=1)
            )
        
        self.bottleneck_block4 = nn.Sequential(
            BottleNeck(in_channel=256, out_channel=384, hidden_channel=96, stride=2),
            BottleNeck(in_channel=384, out_channel=384, hidden_channel=96, stride=1),
            BottleNeck(in_channel=384, out_channel=384, hidden_channel=96, stride=1),
            BottleNeck(in_channel=384, out_channel=384, hidden_channel=96, stride=1),
            BottleNeck(in_channel=384, out_channel=384, hidden_channel=96, stride=1),
            BottleNeck(in_channel=384, out_channel=384, hidden_channel=96, stride=1)
            )
        
        self.bottleneck_block5 = nn.Sequential(
            BottleNeck(in_channel=384, out_channel=512, hidden_channel=128, stride=2),
            BottleNeck(in_channel=512, out_channel=512, hidden_channel=128, stride=1),
            BottleNeck(in_channel=512, out_channel=512, hidden_channel=128, stride=1),
            BottleNeck(in_channel=512, out_channel=512, hidden_channel=128, stride=1),
            BottleNeck(in_channel=512, out_channel=512, hidden_channel=128, stride=1),
            BottleNeck(in_channel=512, out_channel=512, hidden_channel=128, stride=1)
            )
        
        fpn_batchnorm = torch.nn.BatchNorm2d(32)
        self.fpn = torchvision.ops.FeaturePyramidNetwork([128, 192, 256, 384, 512], 32)
        
        self.pooling = SpatialPyramidPooling(in_channels=32, concat_mode=1)
        
    def forward(self, x_lidar, x_map):
        
        x_lidar = self.lidar(x_lidar)
        x_map = self.map(x_map)
        
        print('lidar', x_lidar.size())
        print('map', x_map.size())
        # concat to [96,H,W]
        x = torch.cat((x_lidar, x_map), 1)
        print('cat', x.size())
        
        x_pyramid = OrderedDict()
        
        # bottleneck to [128,H/2,W/2]
        x1 = self.bottleneck_block1(x)
        x_pyramid['x1'] = x1
        
        # bottleneck to [192,H/4,W/4]
        x2 = self.bottleneck_block2(x1)
        x_pyramid['x2'] = x2
        
        # bottleneck to [256,H/8,W/8]
        x3 = self.bottleneck_block3(x2)
        x_pyramid['x3'] = x3
        
        # bottleneck to [384,H/16,W/16]
        x4 = self.bottleneck_block4(x3)
        x_pyramid['x4'] = x4
        
        # bottleneck to [512,H/32,W/32]
        x5 = self.bottleneck_block5(x4)
        x_pyramid['x5'] = x5
        
        x_pyramid = self.fpn(x_pyramid)
        x_pyramid1 = x_pyramid['x1']
        
        x_pooling = self.pooling(x_pyramid1)
        
        return x_pooling

        
        
        
        
class LiDAR(torch.nn.Module):    
    def __init__(self, in_channel):
        super(LiDAR, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        return x
        
    

class Map(torch.nn.Module):
    
    def __init__(self, in_channel):
        super(Map, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        return x
    
    
class BottleNeck(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, hidden_channel, stride=1):
        super(BottleNeck, self).__init__()
        
        self.s2 = stride
        self.conv1 = torch.nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1)
        self.batchnorm_h = torch.nn.BatchNorm2d(hidden_channel)
        self.conv2 = torch.nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=self.s2, padding=1)
        self.conv3 = torch.nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1)
        self.batchnorm_o = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv2d(in_channel, out_channel, 1, self.s2)
        
        
    def forward(self, x):
            
        x1 = self.conv1(x)
        x1 = self.batchnorm_h(x1)
        x1 = self.conv2(x1)
        x1 = self.batchnorm_h(x1)
        x1 = self.conv3(x1)
        x1 = self.batchnorm_o(x1)
        x1 = self.relu(x1)
        
        if self.s2 == 2:
            x2 = self.conv4(x)
            x2 = self.batchnorm_o(x2)
            
            return torch.add(x1, x2)
        
        return x1
        
        
        

if __name__ == '__main__':
    # some quick tests for the SPP module
    c_lidar = 160
    c_map = 3
    x_lidar = torch.randn(2,c_lidar, 100, 100)
    x_map = torch.randn(2,c_map,100,100)

    
    ENCODER = Encoder(c_lidar, c_map)
    print(ENCODER(x_lidar, x_map).size())
    
    
    

        
    
        
        