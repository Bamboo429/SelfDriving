#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:19:15 2024

@author: tracylin
"""

import numpy as np
import torch 
import torch.nn as nn
 
class Decoder(nn.Module):
    
    def __init__(self, in_channel, h1, h2, h3, h4, attn_embed_dim, attn_num_heads):
        
        super(Decoder, self).__init__()
        self.offset = OffsetPrediction(in_channel, h1)
        self.feature = FeatureAggregation(h2, h3, attn_embed_dim, attn_num_heads)
        self.occupancy = OccupancyFlow(in_channel, h3, h4)
        
    def forward(self, Z, Q):
        
        q_z, r = self.offset(Z, Q)
        z = self.feature(Z, Q, q_z, r)
        O, F = self.occupancy(Q, z)
        
        return O, F
 
class OccupancyFlow(nn.Module):
    
    def __init__(self, in_channel, h3, h4):
        
        super(OccupancyFlow, self).__init__()
        self.linear1 = nn.Linear(in_channel+h3, h4)
        self.linear2 = nn.Linear(3, h4)
        self.reslayer = ResidualLayer(h4)
        
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(h4, 1)
        self.linear4 = nn.Linear(h4, 2)
        
        
    def forward(self, Q, z):
        
        # fc
        z = self.linear1(z)
        q = self.linear2(Q)
        of = torch.add(z,q)
        
        # repeat
        for _ in range(3):
            of = self.reslayer(of)
        
        of = self.relu(of)
        
        # output for occupancy and flow
        O = self.linear3(of)
        F = self.linear4(of)
        
        return O, F
        
        
           
class FeatureAggregation(nn.Module):
    
    def __init__(self, h2, h3, attn_embed_dim, attn_num_heads):
        super(FeatureAggregation, self).__init__()
        self.linear1 = nn.Linear(2, h2)
        self.linear2 = nn.Linear(3, h2)
        self.multihead_attn = nn.MultiheadAttention(attn_embed_dim, attn_num_heads, batch_first=True)
        
    def forward(self, Z, Q, q_z, r):
        
        # interpolation
        size_q, size_k, size_xy = r.size()    
        qr = torch.reshape(r, (-1, 2))
        qx = qr[:,0]
        qy = qr[:,1]
        Zr = bilinear_interpolation_3d(Z, qx, qy)
        Zr = torch.reshape(Zr, (size_q, size_k, -1))
        
        # generate key and value by concat
        r = self.linear1(r)
        key = torch.cat((Zr,r),dim=2)
        value = key.clone()
        
        # generate query
        q = self.linear2(Q)
        query = torch.cat((q_z,q),1)
        query = torch.reshape(query, (size_q, 1, -1))
        
        # multihead attention
        attn_output, attn_weight = self.multihead_attn(query, key, value)
        attn_output = torch.reshape(attn_output,(size_q, -1))
        
        # concat attn and q_z
        z = torch.cat((q_z, attn_output),1)
        
        return z
        
class OffsetPrediction(nn.Module):
    
    def __init__(self, in_channel, hidden_channel=16, K=2):
        
        super(OffsetPrediction, self).__init__()
        self.K = K
        
        self.linear1 = nn.Linear(3,hidden_channel)
        self.linear2 = nn.Linear(in_channel, hidden_channel)
        self.reslayer = ResidualLayer(hidden_channel)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_channel, K*2)
        
    
    def forward(self, Z, Q):
        
        q = self.linear1(Q)
        
        # Bilinear interpolation
        qx = Q[:,0]
        qy = Q[:,1]
        q_xy = Q[:,0:2]
        
        Qxy = bilinear_interpolation_3d(Z, qx, qy)
          
        q_z = self.linear2(Qxy)
        
        # residual layer
        q_sum = torch.add(q_z, q)  
        q_sum = self.reslayer(q_sum)
        q_sum = self.relu(q_sum)
        
        # create K offsets
        q_delta = self.linear3(q_sum)
        q_delta = torch.reshape(q_delta, (-1,self.K,2))
        
        # generate r by adding offsets
        q_xy = torch.reshape(q_xy, (-1,1,2))     
        q_xy = q_xy.repeat(1,2,1)
        r = torch.add(q_xy, q_delta)
        
        return Qxy, r
 
class ResidualLayer(nn.Module):
    
    def __init__(self, hidden_channel):
        
        super(ResidualLayer, self).__init__()
        
        self.linear1 = nn.Linear(hidden_channel, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel)
        
    def forward(self,x):
        
        out = self.linear1(x) + x
        out = self.linear2(out)
        
        return out
               
def bilinear_interpolation_3d(Z, x, y):
    
    c = Z.size()[0]
    # Extract grid points
    x1 = x.int()
    y1 = y.int()
    x2 = x1 + 1
    y2 = y1 + 1
    
    # Extract values at the grid points
    Q11 = Z[:, x1, y1]
    Q12 = Z[:, x1, y2]
    Q21 = Z[:, x2, y1]
    Q22 = Z[:, x2, y2]
    
    # Calculate distances
    dx = x - x1
    dy = y - y1
    
    # Perform linear interpolation along x-axis
    R1 = Q11 * (1 - dx) + Q21 * dx
    R2 = Q12 * (1 - dx) + Q22 * dx
    
    # Perform linear interpolation along y-axis
    interpolated_values = R1 * (1 - dy) + R2 * dy
    interpolated_values = torch.reshape(interpolated_values, (-1,c))
    print('interpolated_values:', interpolated_values.size())
    return interpolated_values



if __name__ == '__main__':
    
    c_lidar = 160
    c_map = 8
    Q = torch.randn(10,3)
    Z = torch.randn(c_map,100,100)

    h1 = 16
    h2 = 8
    h3 = c_map+h2
    h4 = 16
    model = Decoder(c_map, h1, h2, h3, h4, h3, 4)
    print(model(Z, Q).size())
