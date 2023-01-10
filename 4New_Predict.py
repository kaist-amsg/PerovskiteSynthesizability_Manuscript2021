import os
import json
from time import time
from sklearn import metrics
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from gcnn.model import GCNN
from gcnn.data import Parallel_Collate_Pool, get_loader, CIFData

from glob import glob
import csv

def main():
    data_path= './data4prediction'
    
    # Best Hyperparameters
    atom_fea_len = 64
    n_conv = 1
    lr_decay_rate = 0.99
    
    #var. for dataset loader
    batch_size = 512
    
    #var for training
    cuda = True
    
    #setup
    print('loading data...',end=''); t = time()
    data = CIFData(data_path,cache_path=data_path)
    print('completed', time()-t,'sec')
    collate_fn = Parallel_Collate_Pool(torch.cuda.device_count(),data.orig_atom_fea_len,data.nbr_fea_len)
    
    loader = get_loader(data,collate_fn,batch_size,[list(range(len(data)))],0,True)[0]
    
    #build model
    model = GCNN(data.orig_atom_fea_len,data.nbr_fea_len,atom_fea_len,n_conv)
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = nn.DataParallel(model)
        model.cuda()
    
    os.makedirs('predict',exist_ok=True)
    '''
    outputs = []
    Bs = []
    for i,p in enumerate(sorted(glob('weights/W0_*'))):
        print('Loading model',p)
        model.load_state_dict(torch.load(p))
        output,target,mpids,B = use_model(loader,model,i)
        outputs.append(output)
        Bs.append(B)
    outputs = np.mean(outputs,axis=0).tolist()
    Bs_mean = []
    for b in zip(*Bs):
        Bs_mean.append(np.mean(b,axis=0).tolist())
    json.dump([mpids,outputs,target,Bs_mean],open('predict/%s.json'%(data_path[3:].replace('/','_')),'w'))
    '''
    outputs = []
    for i,p in enumerate(sorted(glob('weights/*_*'))):
        print('Loading model',p)
        model.load_state_dict(torch.load(p))
        output,target,mpids = use_model(loader,model,i)
        outputs.append(output)
    std = np.std(outputs,axis=0).tolist()
    #json.dump(outputs,open('predict/%s_each_score.json'%(data_path[3:].replace('/','_')),'w'))
    outputs = np.mean(outputs,axis=0).tolist()

    json.dump([mpids,outputs,target,std],open('predict/Perov_All.json','w'))
    
def use_model(data_loader, model, epoch):
    
    batch_time = AverageMeter()
    
    model.eval()
        
    t0 = time()
    outputs = []
    targets = []
    mpids = []
    Bs = []
    for i, (inputs,target,mpid,_) in enumerate(data_loader):
        targets += target.cpu().tolist()
        mpids += mpid
        # move input to cuda
        if next(model.parameters()).is_cuda:
            for j in range(len(inputs)): inputs[j] = inputs[j].to(device='cuda')
            target = target.to(device='cuda')
            
        #compute output
        with torch.no_grad():
            output,Weights = model(*inputs)
        outputs += output.cpu().tolist()
        '''
        B = Weights[0]
        
        # sort atom weight into crystals
        crystal_idx = []
        n = 0 
        for nonpad,idx in zip(inputs[0],inputs[5]):
            idx = idx[:nonpad].cpu().numpy() + n
            crystal_idx.append(idx)
            n = np.max(idx) + 1
        crystal_idx = np.concatenate(crystal_idx)
        
        B = B.cpu().numpy()
        Blist = []
        for j in range(np.max(crystal_idx)+1):
            Blist.append(B[crystal_idx==j].tolist())
        Bs += Blist
        '''
        '''
        # sort edge weight into crystals
        if len(Weights) == 3:
            B = Weights[2]
            edge_idx = []
            n = 0 
            for nonpad,idx in zip(inputs[1],inputs[4]):
                idx = idx[:nonpad,:].cpu().numpy() + n
                edge_idx.append(idx)
                n = np.max(idx) + 1
            edge_idx = np.concatenate(edge_idx)
            
        ''' 
        
        #measure elapsed time
        batch_time.update(time() - t0)
        t0 = time()
        
        s = 'Pred '
        
        print(s+': [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
          epoch, i, len(data_loader), batch_time=batch_time))
    
    print(s+' end: [{0}]\t'
      'Time {batch_time.sum:.3f}'.format(epoch, batch_time=batch_time))
    
    idx = np.argsort(mpids)
    outputs = [outputs[i] for i in idx]
    targets = [targets[i] for i in idx]
    mpids = [mpids[i] for i in idx]
    #Bs = [Bs[i] for i in idx]
    return outputs,targets,mpids
    #return outputs,targets,mpids,Bs
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    # run neural network
    main()
    # compile result
    mpids,outputs,target,std = json.load(open('predict/Perov_All.json','r'))
    with open('prediction.csv', 'w',newline='') as outfile:
        writer = csv.writer(outfile)
        # header
        # number, abc, label, clscore, clstd, sources
        writer.writerow(['id','CL score','CL score std'])
        for cifid,cl,clstd in zip(mpids,outputs,std):
            writer.writerow([cifid,cl,clstd])
        
