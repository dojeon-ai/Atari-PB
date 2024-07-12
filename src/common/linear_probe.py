import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from src.models.heads import MHLinearHead    
    

class ProbeDataset(Dataset):
    def __init__(self, feats, acts, ids):
        self.feats = feats
        self.acts = acts
        self.ids = ids

    def __getitem__(self, index):
        feat = self.feats[index]
        act = self.acts[index].long()
        id = self.ids[index].long()
        
        item = {
            'feat': feat,
            'act': act,
            'id': id            
        }        
        return item

    def __len__(self):
        return len(self.feats)


def probe_action(feats, acts, ids, action_size, device, test_size=0.1, epochs=100):
    print(f'start action probing')
    
    # random shuffle
    idxs = torch.randperm(len(feats))
    feats = feats[idxs]
    acts = acts[idxs]
    ids = ids[idxs]
    
    # train-test split
    split_idx = int(len(feats)*test_size)
    feat_train, act_train, id_train = feats[split_idx:], acts[split_idx:], ids[split_idx:]
    feat_test, act_test, id_test = feats[:split_idx], acts[:split_idx], ids[:split_idx]
    
    train_dataset = ProbeDataset(feat_train, act_train, id_train)
    test_dataset = ProbeDataset(feat_test, act_test, id_test)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512)
    
    # linear model & optimizer
    model = MHLinearHead((feat_train.shape[-1],), action_size, 60).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # train
    model.train()
    for epoch in tqdm.tqdm(range(epochs)):
        for batch in train_dataloader:
            for key, value in batch.items():
                batch[key] = value.to(device)
            
            feat = batch['feat']
            id = batch['id']
            act = batch['act']
            n, t, d = feat.shape
            
            act_pred, _ = model(feat, id)
            act_pred = rearrange(act_pred, 'n t d -> (n t) d')
            act = rearrange(act, 'n t -> (n t)')
            
            optimizer.zero_grad()
            loss = criterion(act_pred, act)
            loss.backward()
            optimizer.step()
            
    # test
    model.eval()
    act_pred_list = []
    for batch in test_dataloader:
        for key, value in batch.items():
            batch[key] = value.to(device)
            
        feat = batch['feat']
        id = batch['id']
        act = batch['act']
        n, t, d = feat.shape
        
        act_pred, _ = model(feat, id)
        act_pred = rearrange(act_pred, 'n t d -> (n t) d')            
        act_pred_list.append(torch.argmax(act_pred,1).cpu().numpy())
    
    act_pred = np.concatenate(act_pred_list)
    act_acc = np.mean(act_pred == act_test.numpy().squeeze())
    
    return act_acc