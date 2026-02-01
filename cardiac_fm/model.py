import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Dense4012FrameRNN, Dense4012FrameRNN_3slices, Three_plus_three_cnn_lstm
from fairseq.fairseq_signals.models import build_model_from_checkpoint

class MMCL(nn.Module):
    def __init__(self, mri_ckpt=None, ecg_ckpt=None):
        super(MMCL, self).__init__()
        self.mri_encoder = Three_plus_three_cnn_lstm()
        if mri_ckpt is not None:
            checkpoint = torch.load(mri_ckpt)
        self.mri_encoder.load_state_dict(checkpoint["model_state_dict"])
        self.num_frames = None
        # Try only training the senc part
        for p in self.mri_encoder.mri_encoder1.fenc.parameters():
            p.requires_grad = False
        for p in self.mri_encoder.mri_encoder2.fenc.parameters():
            p.requires_grad = False   
        
        if ecg_ckpt is None:
            raise ValueError("ECG checkpoint path must be provided.")
        self.ecg_encoder = build_model_from_checkpoint(ecg_ckpt)
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False
        for p in self.ecg_encoder.encoder.parameters(): 
            p.requires_grad = True

        self.ecg_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
        )
        self.mri_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
        )

    def forward_mri(self, mris):
        x1 = mris[:, :, 0:3, :, :]  
        x2 = mris[:, :, 3:6, :, :]  

        x1 = x1.type(torch.FloatTensor)
        x2 = x2.type(torch.FloatTensor)
        h1 = self.mri_encoder.mri_encoder1.init_hidden(x1.size(0))
        mri_features1 = self.mri_encoder.mri_encoder1.embedding(x1, h1)
        h2 = self.mri_encoder.mri_encoder2.init_hidden(x2.size(0))
        mri_features2 = self.mri_encoder.mri_encoder2.embedding(x2, h2)
        mri_features = torch.cat([mri_features1, mri_features2], dim=-1)  
        
        mri_features = self.mri_projection(mri_features) 
        return mri_features
    
    def forward_ecg(self, ecgs):
        ecg_features = self.ecg_encoder.extract_features(source=ecgs["net_input"]["source"], padding_mask=ecgs["net_input"]["padding_mask"])
        x = ecg_features["x"]
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        x = self.ecg_projection(x) 
        return x

    def forward(self, mris, ecgs): 
        mri_proj = self.forward_mri(mris) # shape: 256
        ecg_proj = self.forward_ecg(ecgs) # shape: 768 -> 256
        return F.normalize(mri_proj, dim=-1), F.normalize(ecg_proj, dim=-1)

class CNN_LSTM_model(nn.Module):
    def __init__(self, ckpt=None):
        super(CNN_LSTM_model, self).__init__()
        self.mri_encoder = Three_plus_three_cnn_lstm()
        if ckpt is not None:
            checkpoint = torch.load(ckpt)
            self.mri_encoder.load_state_dict(checkpoint["model_state_dict"])
        self.num_frames = None

        self.pred = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, mris):
        x1 = mris[:, :, 0:3, :, :]  
        x2 = mris[:, :, 3:6, :, :]  

        x1 = x1.type(torch.FloatTensor)
        x2 = x2.type(torch.FloatTensor)

        h1 = self.mri_encoder.mri_encoder1.init_hidden(x1.size(0))
        mri_features1 = self.mri_encoder.mri_encoder1.embedding(x1, h1)
        h2 = self.mri_encoder.mri_encoder2.init_hidden(x2.size(0))
        mri_features2 = self.mri_encoder.mri_encoder2.embedding(x2, h2)
        mri_features = torch.cat([mri_features1, mri_features2], dim=-1)  
        
        pred = self.pred(mri_features) 
        return pred
