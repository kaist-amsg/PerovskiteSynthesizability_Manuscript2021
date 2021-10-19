from __future__ import print_function, division

import csv
import os
    
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.local_env import solid_angle
import pickle
from datetime import datetime
from time import time


def get_loader(dataset, collate_fn=default_collate,
                batch_size=64, idx_sets=None,
                num_workers=1, pin_memory=False):
    loaders = []
    for idx in idx_sets:
        loaders.append(DataLoader(dataset, batch_size=batch_size,
                       sampler=SubsetRandomSampler(idx),
                       num_workers=num_workers,
                       collate_fn=collate_fn, pin_memory=pin_memory))
    return loaders



class Parallel_Collate_Pool(object):
    
    def __init__(self,ngpu,atom_fea_len,nbr_fea_len):
        self.ngpu = ngpu
        self.atom_fea_len=atom_fea_len
        self.nbr_fea_len=nbr_fea_len
    
    def _evenly_split_data(self,dataset_list,ngpu): 
        # distribute data by the number of edge as edge processing is the most demanding
        edgelen = np.array(list(map(lambda x:x[0][1].shape[0],dataset_list)))
        idx = np.argsort(edgelen).tolist()
        split_idx = [[] for _ in range(ngpu)]
        while idx:
            gpuload = [[edgelen[i] for i in j] for j in split_idx]
            gpuload = [np.sum(i) for i in gpuload]
            split_idx[np.argmin(gpuload)].append(idx.pop())
        dataset = [[dataset_list[i] for i in j] for j in split_idx]
        edge_tensor_size = [sum([dataset_list[i][0][1].size(0) for i in j]) for j in split_idx]
        atom_tensor_size = [sum([dataset_list[i][0][0].size(0) for i in j]) for j in split_idx]
        return dataset, atom_tensor_size, edge_tensor_size
    
    def __call__(self,dataset_list):
        """
        Collate a list of data and return a batch for predicting crystal
        properties.
    
        Parameters
        ----------
    
        dataset_list: list of tuples for each data point.
          (atom_fea, edge_fea, edge_fea_idx, target)
    
          atom_fea: torch.Tensor shape (n_i, atom_fea_len)
          edge_fea: torch.Tensor shape (n_i, M, edge_fea_len)
          edge_fea_idx: torch.LongTensor shape (n_i, M)
          target: torch.Tensor shape (1, )
          cif_id: str or int
        ngpu: int. number of gpu for parallizing
        
        Returns
        -------
        N = sum(n_i); N0 = sum(i)
    
        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        batch_edge_fea: torch.Tensor shape (N, M, edge_fea_len)
          Bond features of each atom's M neighbors
        batch_edge_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        target: torch.Tensor shape (N, 1)
          Target value for prediction
        batch_cif_ids: list
        """
        dataset_lists, atom_tensor_sizes, edge_tensor_sizes =\
            self._evenly_split_data(dataset_list,self.ngpu)
       
        batch_atom_fea = []
        batch_edge_fea = []
        batch_edge_fea_idx = []
        batch_crystal_atom_idx = []
        batch_target = []
        batch_cif_ids = []
        batch_unique_atom_idx = []
        for i,dataset in enumerate(dataset_lists):
            base_idx = 0
            gpu_atom_fea = []
            gpu_edge_fea = []
            gpu_edge_fea_idx = []
            gpu_crystal_atom_idx = []
            for j, ((atom_fea, edge_fea, edge_fea_idx), target, cif_id, unique_atom_idx)\
                    in enumerate(dataset):
                n_i = atom_fea.shape[0]  # number of atoms for this crystal
                gpu_atom_fea.append(atom_fea)
                gpu_edge_fea.append(edge_fea)
                gpu_edge_fea_idx.append(edge_fea_idx+base_idx)
                gpu_crystal_atom_idx.append(torch.ones(n_i,dtype=torch.long)*j)
                base_idx += n_i
                batch_target.append(target)
                batch_cif_ids.append(cif_id)
                batch_unique_atom_idx.append(unique_atom_idx)
            gpu_atom_fea.append(torch.zeros(max(atom_tensor_sizes)-atom_tensor_sizes[i],self.atom_fea_len))
            gpu_edge_fea.append(torch.zeros(max(edge_tensor_sizes)-edge_tensor_sizes[i],self.nbr_fea_len))
            gpu_edge_fea_idx.append(torch.zeros(max(edge_tensor_sizes)-edge_tensor_sizes[i],2,dtype=torch.long))
            gpu_crystal_atom_idx.append(torch.zeros(max(atom_tensor_sizes)-atom_tensor_sizes[i],dtype=torch.long))
            
            batch_atom_fea.append(torch.cat(gpu_atom_fea,dim=0))
            batch_edge_fea.append(torch.cat(gpu_edge_fea,dim=0))
            batch_edge_fea_idx.append(torch.cat(gpu_edge_fea_idx,dim=0))
            batch_crystal_atom_idx.append(torch.cat(gpu_crystal_atom_idx,dim=0))
            
        return [torch.LongTensor(atom_tensor_sizes),
                torch.LongTensor(edge_tensor_sizes),
                torch.stack(batch_atom_fea, dim=0),
                torch.stack(batch_edge_fea, dim=0),
                torch.stack(batch_edge_fea_idx, dim=0),
                torch.stack(batch_crystal_atom_idx,dim=0)],\
                torch.stack(batch_target, dim=0),\
                batch_cif_ids,\
                batch_unique_atom_idx


class GaussianExpansion(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, vmin, vmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert vmin < vmax
        assert vmax - vmin > step
        self.vmin = vmin
        self.vmax = vmax
        self.filter = np.arange(vmin, vmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, v):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(v[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

'''
def _process_distcut(inputs):
    cif_id, target, root_dir, oh, gdf, gtf = inputs
    target = np.array([float(target)])
    # loading
    crystal = Structure.from_file(os.path.join(root_dir,cif_id+'.cif'))
    # atom features
    atom_fea = np.vstack([oh[crystal[i].specie.number]
                          for i in range(len(crystal))])
    # neighbor
    all_nbrs = crystal.get_all_neighbors(gdf.dmax, include_index=True,include_image=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for i,nbr in enumerate(all_nbrs):
        for _,d,j,_ in nbr:
            nbr_fea_idx.append([i,j])
            nbr_fea.append(d)
    nbr_fea = np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)
    nbr_fea_idx = np.array(nbr_fea_idx)
    # unique atom index
    uniqueatom_idx = [i[0] for i in SpacegroupAnalyzer(crystal).get_symmetrized_structure().equivalent_indices]
    return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, uniqueatom_idx
'''
def _process_voronoi(inputs):
    cif_id, target, root_dir, oh, gdf, gtf = inputs
    target = np.array([float(target)])
    # loading
    crystal = Structure.from_file(os.path.join(root_dir,cif_id+'.cif'))
    # atom features
    atom_fea = np.vstack([oh[crystal[i].specie.number]
                          for i in range(len(crystal))])
    # neighbor
    mindist = gdf.vmax + 4.0
    while True: # Voroni polyhedra does not form if too small
        try:
            vnn = VoronoiNN(cutoff=mindist,allow_pathological=True,compute_adj_neighbors=False)
            all_nbrs = vnn.get_all_nn_info(crystal)
            break
        except:
            mindist += 4.0
    
    all_nbrs = [sorted(nbrs, key=lambda x: x['poly_info']['face_dist']) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea_d, nbr_fea_t = [], [], []
    for i,nbrs in enumerate(all_nbrs):
        for nbr_info in nbrs: # newer version
            if nbr_info['poly_info']['face_dist']*2 <= gdf.vmax:
                nbr_fea_idx.append([i,nbr_info['site_index']])
                nbr_fea_d.append(nbr_info['poly_info']['face_dist']*2)
                nbr_fea_t.append(nbr_info['poly_info']['solid_angle'])
    nbr_fea_d = gdf.expand(np.array(nbr_fea_d))
    nbr_fea_t = gtf.expand(np.array(nbr_fea_t))
    nbr_fea_idx = np.array(nbr_fea_idx)
    nbr_fea = np.concatenate((nbr_fea_d,nbr_fea_t),axis=1)
    
    # unique atom index
    #uniqueatom_idx = [i[0] for i in SpacegroupAnalyzer(crystal).get_symmetrized_structure().equivalent_indices]
    uniqueatom_idx = []
    return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, uniqueatom_idx

class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, radius=7.0, dmin=0.0, dstep=0.2, tmin=0, tmax=np.pi, tstep=0.2,cache_path='./data_cache'):
        # properties
        self.orig_atom_fea_len = 120
        self.nbr_fea_len = round((radius+dstep-dmin)/dstep) + round((tmax+tstep-tmin)/tstep)
        
        # Preprocess
        os.makedirs(cache_path,exist_ok=True)
        if not os.path.exists(os.path.join(cache_path,'data_cache.pkl')):
            print('Preprocessing data...')
            # assert
            assert os.path.exists(root_dir), 'root_dir does not exist!'
            id_prop_file = os.path.join(root_dir, 'id_prop.csv')
            assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
            
            # define featurization methods
            oh = [[1 if i == j else 0 for j in range(0,self.orig_atom_fea_len)] for i in range(0,self.orig_atom_fea_len)]
            gdf = GaussianExpansion(vmin=dmin, vmax=radius, step=dstep)
            gtf = GaussianExpansion(vmin=tmin, vmax=tmax, step=tstep)
            
            # importing parallel processing
            import random
            try: 
                import multiprocessing
            except:
                multiprocessing = None
            
            # load property file
            with open(id_prop_file) as f:
                reader = csv.reader(f)
                id_prop_data = sorted([row for row in reader],key=lambda x:x[0])
            
            # create generator
            inputs = [[d[0],d[1],root_dir,oh,gdf,gtf] for d in id_prop_data]
            random.shuffle(inputs) # faster for processing sorted mp data.
            if multiprocessing is not None:
                p = multiprocessing.Pool()
                gen = p.imap_unordered(_process_voronoi,inputs)
            else:
                from itertools import imap
                gen = imap(_process_voronoi,inputs)
            
            # loop
            print(datetime.now().strftime("[%H:%M:%S]"),'0','/',len(id_prop_data))
            self.data = []; t = time()
            for i,r in enumerate(gen):
                self.data.append(r)
                if i != 0 and i %1000 == 0:
                    print(datetime.now().strftime("[%H:%M:%S]"),i,'/',len(id_prop_data),\
                    '| %.2f/1000 sec/cif |'%((time()-t)/i*1000), '~%.2f sec left'%((len(id_prop_data)-i)/i*(time()-t)))
                    
            
            # sort
            self.data = sorted(self.data,key=lambda x:x[2])
            
            # dump
            print('Saving Cache...')
            pickle.dump(self.data,open(os.path.join(cache_path,'data_cache.pkl'),'wb'))
        else:
            self.data = pickle.load(open(os.path.join(cache_path,'data_cache.pkl'),'rb'))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, uniqueatom_idx = self.data[idx]
        return (torch.Tensor(atom_fea), torch.Tensor(nbr_fea), torch.LongTensor(nbr_fea_idx)), torch.Tensor(target).squeeze(0), cif_id, uniqueatom_idx
        