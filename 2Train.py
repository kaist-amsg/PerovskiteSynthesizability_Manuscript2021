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
from glob import glob

from gcnn.model import GCNN 

from gcnn.data import Parallel_Collate_Pool, get_loader, CIFData

def split_loader(split_path,idx):
    with open(split_path) as f:
        splits = json.load(f)
        
    testpos_idx = splits[idx]['PTest']
    testneg_idx = splits[idx]['NTest']
    val_idx = splits[idx]['PVal'] + splits[idx]['NVal']
    train_idx = splits[idx]['PTrain']+splits[idx]['NTrain']
    return train_idx,val_idx,testpos_idx, testneg_idx



def main(split_idx):
    # CO BE Model best hyperparameters
    data_path='data'
    name = 'Perov_All'
    split_path = './splits_%s.json'%name
    # Best Hyperparameters
    atom_fea_len = 64
    n_conv = 1
    lr_decay_rate = 0.99
    
    #var. for dataset loader
    batch_size = 512
    
    #var for model
    lr = 0.001
    weight_decay = 0

    #var for training
    epochs = 50
    cuda = True
    
    #setup
    print('loading data...',end=''); t = time()
    data = CIFData(data_path,cache_path=data_path)
    print('completed', time()-t,'sec')
    #collate_fn = Parallel_Collate_Pool(torch.cuda.device_count(),data.orig_atom_fea_len,data.nbr_fea_len)
    collate_fn = Parallel_Collate_Pool(1,data.orig_atom_fea_len,data.nbr_fea_len)
    
    train_idx,val_idx,testpos_idx,testneg_idx = split_loader(split_path,split_idx)
    train_loader, val_loader, testpos_loader, testneg_loader = get_loader(data,
        collate_fn,batch_size,[train_idx,val_idx,testpos_idx,testneg_idx],0,True)
    
    #build model
    model = GCNN(data.orig_atom_fea_len,data.nbr_fea_len,atom_fea_len,n_conv)
    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model,device_ids=[0])
        model.cuda()
    
    ## Transfer learning
    # fix embedding and first layer
    # edge_bn1_1, edge_bn1_2, edge_conv1, edge_bn1_3, atom_bn1_1, atom_bn1_2, atom_conv1, atom_bn1_3
    for l in list(list(list(list(model.children())[0].children())[0].children())[0].children())[1:9]:
        for p in l.parameters():
            p.requires_grad = False

    # atom_embedding, edge_embedding
    for p in list(list(model.children())[0].children())[1:3]:
        for p in l.parameters():
            p.requires_grad = False
    # load model parameters
    model.load_state_dict(torch.load('./base_weights/W0_%03d.pth.tar'%(split_idx)))
    
    ## Training
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
    #scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
    os.makedirs('weights',exist_ok=True); 
    bestval_auc = 0.0
    t0 = time()
    for epoch in range(epochs):
        output,target,_ = use_model(train_loader,model,criterion,optimizer,epoch,'train')
        print('Train AUC score [%d]:'%epoch, metrics.roc_auc_score(target, output))
        output,target,_ = use_model(val_loader,model,criterion,optimizer,epoch,'predict','Val')
        Val_AUC = metrics.roc_auc_score(target, output).tolist()
        print('Val AUC score [%d]:'%epoch, Val_AUC, end=' ')
        if Val_AUC > bestval_auc:
            bestval_auc = Val_AUC
            print('<-Best')
            torch.save(model.state_dict(),'weights/W_%s_%03d.pth.tar'%(name,split_idx))
        else: print('')
        #scheduler.step()

    print('--------Training time in sec-------------')
    print(time()-t0)
    print('Testing. Loading best model')
    model.load_state_dict(torch.load('weights/W_%s_%03d.pth.tar'%(name,split_idx)))
    output1,target1,mpids = use_model(testpos_loader,model,criterion,optimizer,epoch,'predict','Positive')
    output2,target2,_ = use_model(testneg_loader,model,criterion,optimizer,epoch,'predict','Negative')
    print('Predict AUC score:', metrics.roc_auc_score(target1+target2, output1+output2))
    
    # save test result
    os.makedirs('tests',exist_ok=True); 
    idx = np.argsort(mpids)
    testoutput = np.array(output1)[idx].tolist()
    json.dump(testoutput,open('tests/Y_%s_%03d.json'%(name,split_idx),'w'))
    
    
    
    
    
def use_model(data_loader, model, criterion, optimizer, epoch, mode, name = None):
    assert mode in ['train','predict']
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_errors = AverageMeter()
    
    #switch to train model
    if mode == 'train':
        model.train()
    elif mode == 'predict':
        model.eval()
        
    t0 = time()
    outputs = []
    targets = []
    mpids = []
    for i, (inputs,target,mpid,_) in enumerate(data_loader):
        targets += target.cpu().tolist()
        mpids += mpid
        # move input to cuda
        if next(model.parameters()).is_cuda:
            for j in range(len(inputs)): inputs[j] = inputs[j].to(device='cuda')
            target = target.to(device='cuda')
            
        #compute output
        if mode == 'train':
            output,_ = model(*inputs)
            outputs += output.detach().cpu().tolist()
        elif mode == 'predict':
            with torch.no_grad():
                output,_ = model(*inputs)
            outputs += output.cpu().tolist()
        
        
        loss = criterion(output, target)
        
        #measure accuracy
        losses.update(loss.data.cpu().item(), target.size(0))
        acc_errors.update(torch.mean(torch.round(output)==target,dtype=torch.float).cpu(), target.size(0))
        
        if mode == 'train':
            #backward operation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #measure elapsed time
        batch_time.update(time() - t0)
        t0 = time()
        
        if mode == 'train':
            s = 'Epoch'
        else:
            s = 'Pred '
        
        if name is not None:
            s += ' '+ name
        
        print(s+': [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'ACC {acc_errors.val:.3f} ({acc_errors.avg:.3f})'.format(
          epoch, i, len(data_loader), batch_time=batch_time,
          loss=losses, acc_errors=acc_errors))
    print(s+' end: [{0}]\t'
      'Time {batch_time.sum:.3f}\t'
      'Loss {loss.avg:.4f}\t'
      'ACC {acc_errors.avg:.3f}'.format(
      epoch, batch_time=batch_time,
      loss=losses, acc_errors=acc_errors))
    return outputs,targets,mpids
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
    #for i in range(100):
    #    main(i)
    # get ensemble test 
    scores = []
    for p in glob('./tests/*.json'):
        score = json.load(open(p))
        scores.append(score)
        ndata = len(score)

    print('Ensemble Test Score %5.1f %%'%(np.mean(np.mean(scores,axis=0)>0.5)*100))
