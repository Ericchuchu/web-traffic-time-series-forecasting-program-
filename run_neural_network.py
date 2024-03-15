import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_preprocess import TrainDataset_sequence,CFG,train,valid
from model import Pred_Sequence_Model,model_params
from train import train_fn,valid_fn
from sklearn.model_selection import train_test_split


train_dataset = TrainDataset_sequence(train, CFG.train_seq_len, CFG.predict_seq_len)
del train
valid_dataset = TrainDataset_sequence(valid, CFG.train_seq_len, CFG.predict_seq_len)
del valid
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False)

model = Pred_Sequence_Model(params=model_params)
model.to(CFG.device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay= 0.005)
criterion_pred_seq = nn.MSELoss()
criterion_classififcation = nn.CrossEntropyLoss()
best_score = 0

for epoch in range(CFG.epochs):
    start_time = time.time()
    train_loss = train_fn(train_loader, model, criterion_pred_seq, optimizer, epoch,device=CFG.device)
    valid_loss, accuracy= valid_fn(valid_loader, model, criterion_pred_seq, CFG.device)  
    elapsed = time.time() - start_time
    print(
        f" Epoch {epoch+1} - avg_train_loss: {train_loss.avg:.4f}  avg_val_loss:"
        f" {valid_loss.avg:.4f}  time: {elapsed:.0f}s accuracy: {accuracy}"
    )
        
        # 如果需要保存模型
    if accuracy > best_score:
        torch.save(model.state_dict(),'model_state_predict_sequence.pth')
        best_score = accuracy
        
torch.cuda.empty_cache()
gc.collect()