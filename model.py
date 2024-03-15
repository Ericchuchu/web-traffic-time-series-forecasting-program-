import torch.nn as nn

# model paramsters
model_params = {
    'input_dim': 4,
    'lstm_dim': 128,
    'lstm_layer': 6,
    'dense_dim': 128,
    'logit_dim': 1,
    'classification_output_dim': 2,
    'sequence_output_dim' : 64
}
# input 為 (batch_size, 序列長度，特徵數量)
class Classification(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(params['input_dim'], params['dense_dim'] // 2),
            nn.ReLU(),
            nn.Linear(params['dense_dim'] // 2, params['dense_dim']),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(params['dense_dim'], params['lstm_dim'], params['lstm_layer'], batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(params['lstm_dim'] * 2, params['logit_dim'])           
        )
        self.output = nn.Sequential(
            nn.Linear(params['lstm_dim'], params['lstm_dim'] // 2),
            nn.Linear(params['lstm_dim'] // 2, params['classification_output_dim']),      
            )
        
    def forward(self, x):
        x = self.mlp(x)     # torch.Size([1, 128, 128])
        x, _ = self.lstm(x) # torch.Size([1, 128, 256])
        x = self.logits(x)  # torch.Size([1, 128, 1])
        x = x.squeeze(-1)   # torch.Size([1, 128])
        pred = self.output(x)   # torch.Size([1, 2])
        return pred

class Pred_Sequence_Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(params['input_dim'], params['dense_dim'] // 2),
            nn.ReLU(),
            nn.Linear(params['dense_dim'] // 2, params['dense_dim']),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(params['dense_dim'], params['lstm_dim'], params['lstm_layer'], batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(params['lstm_dim'] * 2, params['logit_dim'])           
        )
        self.output = nn.Sequential(
            nn.Linear(params['lstm_dim'], params['sequence_output_dim'] )      
            )
        
    def forward(self, x):
        x = self.mlp(x)     # torch.Size([1, 128, 128])
        x, _ = self.lstm(x) # torch.Size([1, 128, 256])
        x = self.logits(x)  # torch.Size([1, 128, 1])
        x = x.squeeze(-1)   # torch.Size([1, 128])
        pred = self.output(x)   # torch.Size([1, 64])
        return pred


  
