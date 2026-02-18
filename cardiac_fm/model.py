import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Dense4012FrameRNN, Dense4012FrameRNN_3slices, Three_plus_three_cnn_lstm
from fairseq_signals.models import build_model_from_checkpoint

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

class ECGFM(nn.Module):

    """
    Args:
        num_labels (int): Number of output classes (e.g., for classification tasks on expert derived features).
        embed_dim (int, optional): Dimension of the input ECG feature embeddings. Default: 768.
        hidden_dim (int, optional): Dimension of the hidden layer in MLP. Default: 512. (changed from 256 to 512 after 3+3)
        dropout_rate (float, optional): Dropout probability to prevent overfitting. Default: 0.2.
        label_weights (bool, optional): Whether to apply label weighting in the loss function. Default: False.
    """
    
    def __init__(self, ecgfm_ckpt, num_labels=1, embed_dim=768, hidden_dim=512, dropout_rate=0.2, use_label_weights = False):
        super(ECGFM, self).__init__()
        
        self.ecg_encoder = build_model_from_checkpoint(ecgfm_ckpt)
        self.pred = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, ecgs):
        ecg_features = self.ecg_encoder.extract_features(source=ecgs["net_input"]["source"],padding_mask=ecgs["net_input"]["padding_mask"])
        x = ecg_features["x"]
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        x = self.pred(x) 
        
        return x

class CARDIACFM_ECG(nn.Module):

    """
    CARDIACFM ECG-only model
    Args:
        num_labels (int): Number of output classes (e.g., for classification tasks on expert derived features).
        embed_dim (int, optional): Dimension of the input ECG feature embeddings. Default: 768.
        hidden_dim (int, optional): Dimension of the hidden layer in MLP. Default: 512. (changed from 256 to 512 after 3+3)
        dropout_rate (float, optional): Dropout probability to prevent overfitting. Default: 0.2.
        label_weights (bool, optional): Whether to apply label weighting in the loss function. Default: False.
    """
    
    def __init__(self, ecgfm_ckpt, training_type="full",
                 num_labels=1, embed_dim=768, hidden_dim=512, dropout_rate=0.2, use_label_weights = False, cardiacfm_pretrained_ckpt = None):
        super(CARDIACFM_ECG, self).__init__()
        self.ecg_encoder_multi = build_model_from_checkpoint(ecgfm_ckpt)

        self.ecg_projection_multi = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, hidden_dim),
        )
        if cardiacfm_pretrained_ckpt != None:
            with open(cardiacfm_pretrained_ckpt, "rb") as f:
                state = torch.load(f, map_location = torch.device("cpu"))
                state = {k.replace("module.", ""): v for k, v in state.items()}
                state_projection = {k.replace("ecg_projection.", ""): v for k, v in state.items() if "ecg_projection." in k}
                state = {k.replace("ecg_encoder.", ""): v for k, v in state.items() if "ecg_encoder." in k}
        
            self.ecg_encoder_multi.load_state_dict(state, strict=True)
            self.ecg_projection_multi.load_state_dict(state_projection, strict=True)
        
        self.pred = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, ecgs):
        ecg_features = self.ecg_encoder_multi.extract_features(source=ecgs["net_input"]["source"],padding_mask=ecgs["net_input"]["padding_mask"])
        x = ecg_features["x"]
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        x = self.ecg_projection_multi(x)  
        x = self.pred(x) 
        
        return x


class CARDIACFM_ECGMRI(nn.Module):

    """
    CARDIACFM ECGMRI downstream model
    Args:
        num_labels (int): Number of output classes (e.g., for classification tasks on expert derived features).
        embed_dim (int, optional): Dimension of the input ECG feature embeddings. Default: 768.
        hidden_dim (int, optional): Dimension of the hidden layer in MLP. Default: 512. (changed from 256 to 512 after 3+3)
        dropout_rate (float, optional): Dropout probability to prevent overfitting. Default: 0.2.
        label_weights (bool, optional): Whether to apply label weighting in the loss function. Default: False.
    """
    
    def __init__(self, ecgfm_ckpt, device, 
                 num_labels=1, ecg_embed_dim=768, mri_embed_dim = 512, hidden_dim=512, dropout_rate=0.2, use_label_weights = False, cardiacfm_pretrained_ckpt = None):
        super(CARDIACFM_ECGMRI, self).__init__()

        self.device = device

        ############## ECG part ############## 

        self.ecg_encoder_multi = build_model_from_checkpoint(ecgfm_ckpt)

        self.ecg_projection_multi = nn.Sequential(
            nn.LayerNorm(ecg_embed_dim),
            nn.Dropout(0.1),
            nn.Linear(ecg_embed_dim, hidden_dim),
        )
       
        if cardiacfm_pretrained_ckpt != None:
            with open(cardiacfm_pretrained_ckpt, "rb") as f:
                state = torch.load(f, map_location = torch.device("cpu"))
                state = {k.replace("module.", ""): v for k, v in state.items()}
                state_projection = {k.replace("ecg_projection.", ""): v for k, v in state.items() if "ecg_projection." in k}
                state = {k.replace("ecg_encoder.", ""): v for k, v in state.items() if "ecg_encoder." in k}
        
            self.ecg_encoder_multi.load_state_dict(state, strict=True)
            self.ecg_projection_multi.load_state_dict(state_projection, strict=True)
        
        ############## MRI part ############## 

        self.mri_encoder_multi = Three_plus_three_cnn_lstm(device)
        self.mri_projection_multi = nn.Sequential(
            nn.LayerNorm(mri_embed_dim),
            nn.Dropout(0.1),
            nn.Linear(mri_embed_dim, hidden_dim),
        )

        if cardiacfm_pretrained_ckpt != None:
            with open(cardiacfm_pretrained_ckpt, "rb") as f:
                state = torch.load(f, map_location = device)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                state_projection = {k.replace("mri_projection.", ""): v for k, v in state.items() if "mri_projection." in k}
                state = {k.replace("mri_encoder.", ""): v for k, v in state.items() if "mri_encoder." in k}

            self.mri_projection_multi.load_state_dict(state_projection, strict=True)
            self.mri_encoder_multi.load_state_dict(state, strict=True)

        for p in self.mri_encoder_multi.mri_encoder1.fenc.parameters():
            p.requires_grad = False
        for p in self.mri_encoder_multi.mri_encoder2.fenc.parameters():
            p.requires_grad = False 

        ############## Joint part ##############  
        self.pred = nn.Linear(hidden_dim*2, 1)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward_mri_multi(self, mris):
        x1 = mris[:, :, 0:3, :, :]  
        x2 = mris[:, :, 3:6, :, :]  

        x1 = x1.type(torch.FloatTensor)
        x2 = x2.type(torch.FloatTensor)
        # if self.num_frames is not None:
        #     y = np.repeat(y, self.num_frames)
        h1 = self.mri_encoder_multi.mri_encoder1.init_hidden(x1.size(0))
        mri_features1_multi = self.mri_encoder_multi.mri_encoder1.embedding(x1, h1)
        h2 = self.mri_encoder_multi.mri_encoder2.init_hidden(x2.size(0))
        mri_features2_multi = self.mri_encoder_multi.mri_encoder2.embedding(x2, h2)
        mri_features_multi = torch.cat([mri_features1_multi, mri_features2_multi], dim=-1)  
        return mri_features_multi
    
    def forward(self, ecgs, mris):
        ############ ECG Part ############## 
        ecg_features_multi = self.ecg_encoder_multi.extract_features(source=ecgs["net_input"]["source"],padding_mask=ecgs["net_input"]["padding_mask"])
        ecg_multi = ecg_features_multi["x"]
        ecg_multi = torch.div(ecg_multi.sum(dim=1), (ecg_multi != 0).sum(dim=1))
        ecg_multi = self.ecg_projection_multi(ecg_multi) 
        
        ############## MRI part ############## 
        mri_multi = self.forward_mri_multi(mris)
        mri_multi = self.mri_projection_multi(mri_multi)

        ############## Joint part ##############  
        x = torch.cat((ecg_multi, mri_multi), dim=1)
        x = self.pred(x)
        
        return x



