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

import wandb
wandb.init(project="self-driving")

def train(train_dataloader, encoder, decoder, device, 
          n_epochs=20, learning_rate=0.001, print_every=2, plot_every=100):
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_acc_total = 0

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=0.001)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=0.01)
    
    loss_fn = occupancyflow_loss()
    
    for epoch in range(1, n_epochs + 1):
        print('==========epoch ', epoch, '===========')
        acc, loss = train_epoch(train_dataloader, encoder, decoder, 
                           encoder_optimizer, decoder_optimizer, 
                           loss_fn, device)
        print_loss_total += loss
        plot_loss_total += loss
        print_acc_total += acc
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_acc_avg = print_acc_total / print_every
            print('loss;', print_loss_avg)
            print('accuracy:', acc, '%')
            
            wandb.log({'epcsh':epoch, 'loss': print_loss_avg, 'acc':print_acc_avg})
            
            print_loss_total = 0
            print_acc_total = 0
            
    
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, 
                decoder_optimizer, loss_fn, device):
 
    total_loss = 0
    total_acc = 0
    for data in dataloader:
        
        # get data from dataloader
        L, M, Q, O, F = data
              
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # data to gpu
        L = L.to(device)
        M = M.to(device)
        Q = Q.to(device)
        O = O.to(device)
        F = F.to(device)
        #print(L.size(), M.size(), Q.size(), O.size(), F.size())
        
        
        # optimizer init
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # get feature map(Z) from encoder
        
        Z = encoder(L, M)
        #print('Z:',Z)
        # output size init
        num_b, num_q, num_c = Q.size()
        O_output = torch.empty(num_b, num_q, 1).to(device)
        F_output = torch.empty(num_b, num_q, 2).to(device)
        
        # decoder 
        for i, (z, q) in enumerate(zip(Z,Q)):
            #print('batch:', i, z, q)
            o, f = decoder(z, q)
            #print('o',o)
            #o_nor = torch.nn.Sigmoid()
            #o = o_nor(o)
            #print(o)
            #o = torch.unsqueeze(o, 0)
            #f = torch.unsqueeze(f, 0)
            
            O_output[i,:], F_output[i,:] = o, f
                        
            '''
            if O_output is None :
                O_output, F_output = o, f
            else:
                O_output = torch.cat((O_output, o), dim=0)
                F_output = torch.cat((F_output, f), dim=0)
            '''
        
        #
          
        # calculate loss
        occupancy_loss_fn, flow_loss_fn, lamda = loss_fn
        
        #O = torch.reshape(O, (b,-1))
        #O_output = torch.reshape(O_output, (b,-1))
      
        occupancy_loss = occupancy_loss_fn(O_output, O)
        flow_loss = flow_loss_fn(F_output, F)
        
        
        #print('O:', O)
        #print('O-output:', O_output)
        #print('o_loss:', occupancy_loss.item())
        #print('f_loss:', flow_loss.item())
        
        
        loss = occupancy_loss+lamda*flow_loss
       
        # back propogation
        flow_loss.backward(retain_graph=True) 
        occupancy_loss.backward()           
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        # calculate loss
        total_loss += loss.item()
        
        # calculate accuracy
        o_nor = torch.nn.Sigmoid()
        O_prob = o_nor(O_output)
        O_prob[O_prob>=0.5] = 1
        O_prob[O_prob<0.5] = 0
        acc = 100*(torch.sum(O_prob==O).item()/num_q)
        
        total_acc += acc 
        #print('accuracy:', acc)
        
    return total_acc/len(dataloader), total_loss / len(dataloader)

def occupancyflow_loss():
    
    occupancy_loss = nn.BCEWithLogitsLoss() #nn.BCELoss() #
    flow_loss = nn.MSELoss()
    flow_lamda = 0.1
    
    return occupancy_loss, flow_loss, flow_lamda


class RandomDataset(Dataset):
    def __init__(self, c_lidar, c_map, q_size, H, W):
        self.c_lidar = c_lidar
        self.c_map = c_map
        self.q_size = q_size
        self.H = H
        self.W = W
        
        self.L = torch.randn(self.c_lidar, self.H, self.W)
        self.M = torch.randn(self.c_map, self.H, self.W)
        self.Q = torch.randn(self.q_size,3)
        
        self.O = torch.randint(0,2,(self.q_size,1),dtype=torch.float) #torch.randn(self.q_size,1) #
        self.F = torch.randn(self.q_size,2)
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        '''
        L = torch.randn(self.c_lidar, self.H, self.W)
        M = torch.randn(self.c_map, self.H, self.W)
        Q = torch.randn(self.q_size,3)
        O = torch.rand(self.q_size,1)
        F = torch.randn(self.q_size,2)
        '''
        return self.L, self.M, self.Q, self.O, self.F
        #return L, M, Q, O, F
    


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    torch.manual_seed(0)
    
    c_lidar = 500
    c_map = 3
    q_size = 50
    H = 100
    W = 100
    attn_num_heads = 4
    
    training_data = RandomDataset(c_lidar, c_map, q_size, H, W)
    train_dataloader = DataLoader(training_data, batch_size=20, shuffle=True)   
    
    h1 = 16
    h2 = 8
    h3 = 64+h2
    h4 = 16
    
    encoder = Encoder(c_lidar, c_map).to(device)
    decoder = Decoder(64, h1, h2, h3, h4, attn_num_heads).to(device)
    
    
    
    train(train_dataloader, encoder, decoder, device,
          n_epochs = 100, print_every=1, plot_every=5)
    
    