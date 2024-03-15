import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest



# setup configuration
class CFG:
    M_quantiles = 50
    attack_percentage = 0.05
    train_seq_len = 128
    predict_seq_len = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    epochs = 40
    learning_rate = 4e-3

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# prepare data
data = pd.read_csv("train_1.csv.zip",encoding = 'utf-8', compression = "zip", encoding_errors = 'ignore')
data = data.sample(5000)
# 缺失值轉為median
data.iloc[:, 1:551] = data.iloc[:, 1:551].fillna(data.iloc[:, 1:551].median())
data.iloc[:, 1:551] = data.iloc[:, 1:551].fillna(0)

'''# 計算每個網頁的traffic
total_traffic = data.iloc[:,1:].sum(axis=1)

# 計算分位數
quantiles = np.quantile(total_traffic, np.linspace(0, 1, CFG.M_quantiles+1))
quantiles = quantiles[1:-1]
page_quantile = []
total_traffic = np.array(total_traffic)

# 使用 np.digitize 將每個 traffic 值分配到對應的分位數區間
page_quantile = np.digitize(total_traffic, quantiles, right=False)+1
data['quantile'] = page_quantile
fifty_percentage_total_traffic = data.iloc[:,1:276].sum(axis=1)
A_noise = np.abs(np.random.normal( fifty_percentage_total_traffic.mean()*CFG.attack_percentage, 
                            fifty_percentage_total_traffic.std()*CFG.attack_percentage, 
                            (CFG.M_quantiles,100)))

# 前275個time series設為train sequence,後220個time series作為攻擊後的label,在後55個time series作為無攻擊的label
# make sequence Sk
medians = data.iloc[:, 1:551].median(axis=1)
data['medians'] = medians
total_median = data.groupby('quantile')['medians'].sum()
total_median = total_median.reindex(range(1, CFG.M_quantiles + 1), fill_value=0)

def update_row(row, total_median, A_noise, CFG):
    quantile = row['quantile']
    median = row['medians']  # 假設 medians.name 是中位數列的名稱
    ratio = median / total_median[quantile]
    noise_factor = A_noise[quantile - 1] / (CFG.M_quantiles + 1 - quantile)
    row[data.columns[276:376]] += noise_factor * ratio
    return row

# 應用這個函數到每一行
data = data.apply(update_row, axis=1, args=(total_median, A_noise, CFG))'''

# 寬數據轉換為長數據
class TrainDataset_sequence(Dataset):
    def __init__(self, data, train_seq_len, predict_seq_len):  
        self.df_long = None
        self.page_idxs = None 
        self.data = data
        self.data_chunk = None      
        self.train_seq_len = train_seq_len
        self.predict_seq_len = predict_seq_len
        self.feature_cols = ['visit','project','access','agent']
        self.preprocess()
    
    def __len__(self):
        return len(self.page_idxs)
    
    def __getitem__(self, idx): # 取出要train的sample跟label
        sample = self.df_long.loc[self.df_long['Page'] == self.page_idxs[idx]].reset_index(drop=True)       
        sample_len = len(sample)
    
        start = np.random.randint(sample_len - self.train_seq_len - self.predict_seq_len - 1)  
        end = start + self.train_seq_len
        
        sample[self.feature_cols] = sample[self.feature_cols].fillna(sample[self.feature_cols].median()).fillna(0)
        inputs = torch.tensor(sample.loc[start : end-1, self.feature_cols].values , dtype = torch.float) 
        # print(len(sample.loc[start:end-1]))
        # print(np.sum(np.isnan(inputs)) )
        labels = sample.loc[end: end + self.predict_seq_len - 1, 'visit']
        labels = labels.fillna(labels.median()).fillna(0).values   
        labels = torch.tensor(labels, dtype = torch.float)
        inputs = inputs.to(CFG.device)
        labels = labels.to(CFG.device)
        return inputs, labels
    
    def preprocess(self):
        # data = data.sample(1000).reset_index(drop = True)
        self.df_long = pd.melt(self.data, id_vars=['Page'], value_vars=self.data.columns[1:551], var_name='Date', value_name='visit')        
        self.df_long.sort_values(by=['Page', 'Date'], inplace=True)
        self.df_long['visit'] = self.df_long.groupby('Page')['visit'].transform(lambda x: x.fillna(x.median()))
        # 分配 label
        # self.df_long['label_attack'] = self.df_long.groupby('Page').apply(assign_labels_attack).reset_index(level=0, drop=True)
        # self.df_long['label_nonattack'] = self.df_long.groupby('Page').apply(assign_labels_nonattack).reset_index(level=0, drop=True)
        self.df_long = self.get_cat_cols(self.df_long)
        self.df_long['visit'] = np.log1p(self.df_long['visit']).astype('float32')   
        # print(self.df_long)        
        self.page_idxs = self.data['Page'].unique()    
    
    # 處理數據
    def parse_page(self,page):
        page = page.split('_')
        return ' '.join(page[:-3]), page[-3], page[-2], page[-1]

    # 對label進行分類
    def get_cat_cols(self,data):
        le = LabelEncoder()
        data['name'], data['project'], data['access'], data['agent'] = zip(*data['Page'].apply(self.parse_page))
        data['project'] = le.fit_transform(data['project'])
        data['access'] = le.fit_transform(data['access'])
        data['agent'] = le.fit_transform(data['agent'])
        data['page_id'] = le.fit_transform(data['Page'])
        return data

    def assign_labels_attack(self,group):
        # 分組創造空的 label 列
        labels = pd.Series([0] * len(group), index=group.index)
        labels.iloc[275:275+220] = 1
        return labels

    def assign_labels_nonattack(self,group):
        # 分組創造空的 label 列
        labels = pd.Series([1] * len(group), index=group.index)
        labels.iloc[275:275+220] = 0
        return labels


class TrainDataset_classification(Dataset):
    def __init__(self, df, train_seq_len):  
        self.feature_cols = ['visit','project','access','agent']
        self.df_long = pd.melt(df, id_vars=['Page'], value_vars=df.columns[1:551], var_name='Date', value_name='visit')        
        self.df_long.sort_values(by=['Page', 'Date'], inplace=True)
        self.df_long['visit'] = self.df_long.groupby('Page')['visit'].transform(lambda x: x.fillna(x.median()))
        # 分配 label
        self.df_long = self.get_cat_cols(self.df_long)
        self.df_long['label_attack'] = self.df_long.groupby('Page').apply(self.assign_labels_attack).reset_index(level=0, drop=True)
        self.df_long['visit'] = np.log1p(self.df_long['visit']).astype('float32')   
        print(self.df_long)        
        self.train_seq_len = train_seq_len
        self.page_idxs = df['Page'].unique()    
    
    def __len__(self):
        return len(self.page_idxs)
    
    def __getitem__(self, idx): # 取出要train的sample跟label
        sample = self.df_long.loc[self.df_long['Page'] == self.page_idxs[idx]].reset_index(drop=True)       
        sample_len = len(sample)
    
        start = np.random.randint(sample_len - self.train_seq_len - 1)  
        end = start + self.train_seq_len
        
        sample[self.feature_cols] = sample[self.feature_cols].fillna(sample[self.feature_cols].median()).fillna(0)
        inputs = torch.tensor(sample.loc[start : end-1, self.feature_cols].values , dtype = torch.float) 
        # print(len(sample.loc[start:end-1]))
        # print(np.sum(np.isnan(inputs)) )
        labels_array = sample.loc[end-1, ['label_attack']].values
        labels_array = labels_array.astype(float)  
        labels = torch.tensor(labels_array, dtype=torch.float)
        # print(inputs.shape, labels.shape)
        return inputs, labels

    # 處理數據
    def parse_page(self,page):
        page = page.split('_')
        return ' '.join(page[:-3]), page[-3], page[-2], page[-1]

    # 對label進行分類
    def get_cat_cols(self,data):
        le = LabelEncoder()
        data['name'], data['project'], data['access'], data['agent'] = zip(*data['Page'].apply(self.parse_page))
        data['project'] = le.fit_transform(data['project'])
        data['access'] = le.fit_transform(data['access'])
        data['agent'] = le.fit_transform(data['agent'])
        data['page_id'] = le.fit_transform(data['Page'])
        return data
    
    def assign_labels_attack(self, df_group):
        """
        應用 Isolation Forest 到每個分組的數據上，返回異常標記。
        """
        # 假設每個分組的數據矩陣為 [n_samples, n_features]
        clf = IsolationForest(n_estimators=50, contamination=0.3, random_state=42)
        clf.fit(df_group['visit'].values.reshape(-1, 1))

        # 獲得異常標記並轉換（-1 變為 1 表示有攻擊，1 變為 0 表示無攻擊）
        preds = clf.predict(df_group['visit'].values.reshape(-1, 1))
        attack_labels = pd.Series(preds, index=df_group.index).map({-1: 1, 1: 0})
        return attack_labels
    
# 劃分訓練集，驗證集
train_len = int(0.8 * len(data))
val_len = int(0.2 * len(data))
# Use random_split to split the dataset into two parts
train, valid = train_test_split(data, test_size = 0.2, random_state=42)
# train_dataset = TrainDataset_classification(train, CFG.train_seq_len)
gc.collect()
