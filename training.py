#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:12:25 2024

@author: tracylin
"""
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch import optim
import time

from encoder import Encoder
from decoder import Decoder



def train(train_dataloader, encoder, decoder, n_epochs=20, learning_rate=0.001,
               print_every=100, plot_every=100):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every


    encoder_optimizer = optim.AdamW(encoder.parameters())
    decoder_optimizer = optim.AdamW(decoder.parameters())
    
    loss_fn = occupancyflow_loss()
    
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, 
                decoder_optimizer, loss_fn):
 
    total_loss = 0
    for data in dataloader:
        L, M, Q, O, F = data
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        L = L.to(device)
        M = M.to(device)
        Q = Q.to(device)
        O = O.to(device)
        F = F.to(device)
        #print(L.size(), M.size(), Q.size(), O.size(), F.size())
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        Z = encoder(L, M)
        O_output = None
        for z, q in zip(Z,Q):
            o, f = decoder(z, q)
            o = torch.unsqueeze(o, 0)
            f = torch.unsqueeze(f, 0)
            
            if O_output is None :
                O_output, F_output = o, f
            else:
                O_output = torch.cat((O_output, o), dim=0)
                F_output = torch.cat((F_output, f), dim=0)
        
        print('O_output:', O_output.size())
            

        occupancy_loss_fn, flow_loss_fn, lamda = loss_fn
        occupancy_loss = occupancy_loss_fn(O, O_output)
        flow_loss = flow_loss_fn(F, F_output)
        
        occupancy_loss.backward(retain_graph=True)
        flow_loss.backward()

        loss = occupancy_loss+lamda*flow_loss
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def occupancyflow_loss():
    
    occupancy_loss = nn.CrossEntropyLoss()
    flow_loss = nn.MSELoss()
    flow_lamda = 0.1
    
    return occupancy_loss, flow_loss, flow_lamda


class RandomDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        
        c_lidar = 160
        c_map = 8
        q_size = 10
        L = torch.randn(c_lidar, 100, 100)
        M = torch.randn(c_map,100,100)
        Q = torch.randn(q_size,3)
        O = torch.randn(q_size,1)
        F = torch.randn(q_size,2)
        
        return L, M, Q, O, F
 
    
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_data = RandomDataset()
    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
    
    c_lidar = 160
    c_map = 8
    q_size = 10
    c_z = 64
    
    h1 = 16
    h2 = 8
    h3 = c_z+h2
    h4 = 16
    
    encoder = Encoder(c_lidar, c_map).to(device)
    decoder = Decoder(c_z, h1, h2, h3, h4, 4).to(device)
    
    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)