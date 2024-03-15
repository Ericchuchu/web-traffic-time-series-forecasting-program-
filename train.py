
import time
import warnings
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 計算smape
def SMAPE(true, predicted):
    true_o = true
    pred_o = predicted
    summ = np.abs(true_o) + np.abs(pred_o)/2
    smape = np.where(summ==0, 0, np.abs(pred_o - true_o) / summ)
    smape = smape.sum()/len(true)
    return smape

softmax = torch.nn.Softmax(dim=1)

def calculate_scores(y_true, y_pred):
    # 提取最可能的类别作为预测结果
    y_pred = np.array(y_pred)
    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_labels == y_true)
    return accuracy

def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler=None, device='cuda'):
    model.train()
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    total_steps = len(train_loader)
    with tqdm(total=total_steps, desc=f"Epoch {epoch+1}", leave=True, ncols=100, unit='step') as pbar:
        for step, (inputs, labels) in enumerate(train_loader):
            batch_size = 64
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            labels = labels.squeeze()
            
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses.update(loss.item(), batch_size)
            
            '''print(
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}]"
                f" Elapsed: {(time.time()-start):.0f}s"
                f" Loss: {losses.val:.4f}"          
            )'''

            pbar.update(1)        
    return losses


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    start = time.time()
    preds = []
    true = []
    
    for step, (inputs, labels) in enumerate(valid_loader):
        batch_size = 64
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        labels = labels.squeeze()
        
        with torch.no_grad():
            y_pred = model(inputs)
        
        loss = criterion(y_pred, labels)  
        true.append(labels.to('cpu').numpy())
        preds.append(softmax(y_pred).to('cpu').numpy())
        losses.update(loss.item(), batch_size)

    true = np.concatenate(true)
    preds = np.concatenate(preds)
    accuracy = calculate_scores(true, preds)
    return losses, accuracy 
        