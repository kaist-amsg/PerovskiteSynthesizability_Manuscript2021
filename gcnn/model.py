# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:49:26 2020

@author: user
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_mean, scatter_min

class FilterLayer(nn.Module):
    def __init__(self, in_fea_len,out_fea_len):
        super(FilterLayer, self).__init__()
        self.lin1 = nn.Linear(in_fea_len,2*out_fea_len)
        self.bn1 = nn.BatchNorm1d(2*out_fea_len)
        self.act = nn.Softplus()
        self.lin2 = nn.Linear(out_fea_len,out_fea_len)
        self.bn2 = nn.BatchNorm1d(out_fea_len)        
        self.bn3 = nn.BatchNorm1d(out_fea_len)        
       
    def forward(self, fea):
        filter_core = self.lin1(fea)
        filter_core = self.bn1(filter_core)
        filter,core = filter_core.chunk(2, dim=1)
        out = torch.sigmoid(filter) * self.act(core)
        out = self.bn2(out)
        out = self.act(self.bn2(self.lin2(out)))
        out = self.bn3(out)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_fea_len,out_fea_len):
        super(DenseLayer, self).__init__()
        self.lin1 = nn.Linear(in_fea_len,out_fea_len)
        self.bn1 = nn.BatchNorm1d(out_fea_len)
        self.act = nn.Softplus()     
        self.bn2 = nn.BatchNorm1d(out_fea_len)
        
    def forward(self, fea):
        fea = self.lin1(fea)
        fea = self.bn1(fea)
        fea = self.act(fea)
        #fea = self.bn2(fea) 
        return fea

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, in_atom_fea_len, in_edge_fea_len, fea_len):
        super(ConvLayer, self).__init__()
        self.act = torch.nn.Softplus()
        
        # edge convoltuion
        self.edge_conv1 = DenseLayer(2*in_atom_fea_len+in_edge_fea_len,fea_len)
        self.atom_conv1 = DenseLayer(in_atom_fea_len+fea_len,fea_len)
        #self.edge_conv1 = FilterLayer(2*in_atom_fea_len+in_edge_fea_len,fea_len)
        #self.atom_conv1 = FilterLayer(in_atom_fea_len+fea_len,fea_len)
        
        self.edge_bn1_1 = nn.BatchNorm1d(2*in_atom_fea_len)
        self.edge_bn1_2 = nn.BatchNorm1d(2*in_atom_fea_len+in_edge_fea_len)
        self.edge_bn1_3 = nn.BatchNorm1d(fea_len)
        self.atom_bn1_1 = nn.BatchNorm1d(fea_len)
        self.atom_bn1_2 = nn.BatchNorm1d(in_atom_fea_len+fea_len)
        self.atom_bn1_3 = nn.BatchNorm1d(fea_len)
        
        # attention
        self.WA1 = nn.Linear(fea_len,2*fea_len)
        self.bnA1 = nn.BatchNorm1d(fea_len*2)
        self.bnA2 = nn.BatchNorm1d(fea_len)
        self.WA2 = nn.Linear(fea_len,fea_len)

        # second layer
        # self convolution
        self.edge_conv2 = DenseLayer(fea_len*3,fea_len)
        self.atom_conv2 = DenseLayer(fea_len*2,fea_len)
        #self.edge_conv2 = FilterLayer(fea_len*3,fea_len)
        #self.atom_conv2 = FilterLayer(fea_len*2,fea_len)
        
        self.edge_bn2_1 = nn.BatchNorm1d(2*fea_len)
        self.edge_bn2_2 = nn.BatchNorm1d(3*fea_len)
        self.edge_bn2_3 = nn.BatchNorm1d(fea_len)
        self.atom_bn2_1 = nn.BatchNorm1d(fea_len)
        self.atom_bn2_2 = nn.BatchNorm1d(2*fea_len)
        self.atom_bn2_3 = nn.BatchNorm1d(fea_len)

        # second layer
        # self convolution
        self.edge_conv3 = DenseLayer(fea_len*3,fea_len)
        self.atom_conv3 = DenseLayer(fea_len*2,fea_len)
        #self.edge_conv2 = FilterLayer(fea_len*3,fea_len)
        #self.atom_conv2 = FilterLayer(fea_len*2,fea_len)
        
        self.edge_bn3_1 = nn.BatchNorm1d(2*fea_len)
        self.edge_bn3_2 = nn.BatchNorm1d(3*fea_len)
        self.edge_bn3_3 = nn.BatchNorm1d(fea_len)
        self.atom_bn3_1 = nn.BatchNorm1d(fea_len)
        self.atom_bn3_2 = nn.BatchNorm1d(2*fea_len)
        self.atom_bn3_3 = nn.BatchNorm1d(fea_len)

    def forward(self, atom_fea, edge_fea, edge_idx):
        ### first layer
        ## atom pooling + edge > new edge
        atom_edge_fea = torch.flatten(atom_fea[edge_idx, :],1,2)
        atom_edge_fea = self.edge_bn1_1(atom_edge_fea)
        edge_fea_cat = torch.cat([atom_edge_fea, edge_fea], dim=1)
        edge_fea_cat = self.edge_bn1_2(edge_fea_cat)
        edge_fea = edge_fea + self.edge_conv1(edge_fea_cat) # skip
        edge_fea = self.edge_bn1_3(edge_fea)
        
        ## edge pooling + atom > new atom
        pooled_nbr_fea = scatter_mean(edge_fea,edge_idx[:,0], dim=0,dim_size=atom_fea.size(0))
        pooled_nbr_fea = self.atom_bn1_1(pooled_nbr_fea)
        atom_fea_cat = torch.cat([atom_fea,pooled_nbr_fea],dim=1)
        atom_fea_cat = self.atom_bn1_2(atom_fea_cat)
        atom_fea = atom_fea + self.atom_conv1(atom_fea_cat)
        atom_fea = self.atom_bn1_3(atom_fea)
        
        ### second layer
        ## atom pooling + edge > new edge
        atom_edge_fea = torch.flatten(atom_fea[edge_idx, :],1,2)
        atom_edge_fea = self.edge_bn2_1(atom_edge_fea)
        edge_fea_cat = torch.cat([atom_edge_fea, edge_fea], dim=1)
        edge_fea_cat = self.edge_bn2_2(edge_fea_cat)
        edge_fea = edge_fea + self.edge_conv2(edge_fea_cat)
        edge_fea = self.edge_bn2_3(edge_fea)
        
        ## edge pooling + atom > new atom

        pooled_nbr_fea = scatter_mean(edge_fea,edge_idx[:,0], dim=0,dim_size=atom_fea.size(0))
        pooled_nbr_fea = self.atom_bn2_1(pooled_nbr_fea)
        atom_fea_cat = torch.cat([atom_fea,pooled_nbr_fea],dim=1)
        atom_fea_cat = self.atom_bn2_2(atom_fea_cat)
        atom_fea = atom_fea + self.atom_conv2(atom_fea_cat)
        atom_fea = self.atom_bn2_3(atom_fea)
        
        ### Third layer
        ## atom pooling + edge > new edge
        atom_edge_fea = torch.flatten(atom_fea[edge_idx, :],1,2)
        atom_edge_fea = self.edge_bn3_1(atom_edge_fea)
        edge_fea_cat = torch.cat([atom_edge_fea, edge_fea], dim=1)
        edge_fea_cat = self.edge_bn3_2(edge_fea_cat)
        edge_fea = edge_fea + self.edge_conv3(edge_fea_cat)
        edge_fea = self.edge_bn3_3(edge_fea)
        
        ## edge pooling + atom > new atom
        '''
        pooled_nbr_fea = scatter_mean(edge_fea,edge_idx[:,0], dim=0,dim_size=atom_fea.size(0))
        pooled_nbr_fea = self.atom_bn2_1(pooled_nbr_fea)
        atom_fea_cat = torch.cat([atom_fea,pooled_nbr_fea],dim=1)
        atom_fea_cat = self.atom_bn2_2(atom_fea_cat)
        atom_fea = atom_fea + self.atom_conv2(atom_fea_cat)
        atom_fea = self.atom_bn2_3(atom_fea)
        '''
        return edge_fea


class GCNN(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, orig_nbr_fea_len, fea_len=64, n_conv=3):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(GCNN, self).__init__()
        self.convs = nn.ModuleList([ConvLayer(in_atom_fea_len=fea_len,
                                              in_edge_fea_len=fea_len,
                                              #in_edge_fea_len=orig_nbr_fea_len,
                                              fea_len=fea_len)
                                    for _ in range(n_conv)])
         
        self.atom_embedding = nn.Sequential(nn.Linear(orig_atom_fea_len, fea_len),nn.BatchNorm1d(fea_len),nn.Softplus())
        self.edge_embedding = nn.Sequential(nn.Linear(orig_nbr_fea_len, fea_len),nn.BatchNorm1d(fea_len),nn.Softplus())
        
        
        self.ToWeight = nn.Sequential(
                        nn.Linear(fea_len, fea_len),
                        nn.BatchNorm1d(fea_len),
                        nn.Softplus(),
                        nn.Linear(fea_len, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, atom_nonpad_size, edge_nonpad_size,atom_one_hot, edge_fea, edge_idx, crystal_idx):
        """
        Forward pass
        B: Number of gpu. It will always be 1 for each gpu
        N: Total number of atoms in the batch
        NP: N + padding for parallelization
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        # Remove paddings. This is done for parallelizing the batch over gpus
        atom_nonpad_size = atom_nonpad_size.squeeze(0)
        edge_nonpad_size = edge_nonpad_size.squeeze(0)
        atom_one_hot = atom_one_hot.squeeze(0)[:atom_nonpad_size]
        edge_fea = edge_fea.squeeze(0)[:edge_nonpad_size]
        edge_idx = edge_idx.squeeze(0)[:edge_nonpad_size]
        crystal_idx = crystal_idx.squeeze(0)[:atom_nonpad_size]
        
        # actual processing
        atom_fea = self.atom_embedding(atom_one_hot)
        edge_fea = self.edge_embedding(edge_fea)
        edge_fea = self.convs[0](atom_fea,edge_fea,edge_idx)

        b = self.ToWeight(edge_fea).reshape(-1)
        b = torch.clamp(b,-8,8)
        
        crystal_edge_idx = crystal_idx[edge_idx[:,0]]
        cs = self.sigmoid(scatter_min(b,crystal_edge_idx,dim_size=crystal_idx.max()+1)[0])
        #cs = self.sigmoid(scatter_mean(b,crystal_idx))
        
        ## Extract fingerprint
        '''
        with torch.no_grad(): # this is for visualization purposes so no need to track back propagation
            nbr_onehot = scatter(torch.round(a)*atom_one_hot[edge_idx[:,1],:],
                edge_idx[:,0], dim=0,dim_size=atom_one_hot.size(0)) # multiply one hot of the edge feature and sum
            FP = torch.cat((atom_one_hot,nbr_onehot),dim=1)
        '''
        return cs, (b,)
