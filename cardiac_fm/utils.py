import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np 

from CNN_LSTM.frame import densenet_40_12_bc, densenet_40_12_bc_3slices
from CNN_LSTM.sequence import RNN
from torch.autograd import Variable

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_lr2(optimizer, warmup_length, steps):

    initial_lrs = [group['lr'] for group in optimizer.param_groups]

    def _lr_adjuster(step):
        if step < warmup_length:
            factor = (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            factor = 0.5 * (1 + np.cos(np.pi * e / es))
        for lr0, group in zip(initial_lrs, optimizer.param_groups):
            group['lr'] = lr0 * factor

        return [group['lr'] for group in optimizer.param_groups]

    return _lr_adjuster

def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

class MRISequenceNet(nn.Module):
    """
    Simple container network for MRI sequence classification. This module consists of:

        1) A frame encoder, e.g., a ConvNet/CNN
        2) A sequence encoder for merging frame representations, e.g., an RNN

    """

    def __init__(self, frame_encoder, seq_encoder, use_cuda=False):
        super(MRISequenceNet, self).__init__()
        self.fenc = frame_encoder
        self.senc = seq_encoder
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        return self.senc.init_hidden(batch_size)

    def embedding(self, x, hidden):
        """Get learned representation of MRI sequence"""
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        batch_size, num_frames, num_channels, width, height = x.size()
        x = x.view(-1, num_channels, width, height)
        x = self.fenc(x)
        x = x.view(batch_size, num_frames, -1)
        x = self.senc.embedding(x, hidden)
        return x
        # if self.use_cuda:
        #     return x.cpu()
        # else:
        #     return x

    def forward(self, x, hidden=None):
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        # collapse all frames into new batch = batch_size * num_frames
        batch_size, num_frames, num_channels, width, height = x.size()
        # print('input: ', x.size()) [4, 13, 3, 64, 64]
        x = x.contiguous().view(-1, num_channels, width, height)
        # encode frames
        # print('input_view: ',x.size())   [52, 3, 64, 64]
        x = self.fenc(x)
        # print('fenc output: ',x.size())    [52, 132, 1, 1]
        x = x.view(batch_size, num_frames, -1)
        # print('fenc output view: ',x.size()) [2, 13, 132]
        # encode sequence
        x = self.senc(x, hidden)
        # print('senc output: ',x.size()) [2, 2]
        return x

    def predict_proba(self, data_loader, binary=True, pos_label=1):
        """ Forward inference """
        y_pred = []
        for i, data in enumerate(data_loader):
            x, y = data
            x = Variable(x) if not self.use_cuda else Variable(x).cuda()
            y = Variable(y) if not self.use_cuda else Variable(y).cuda()
            h0 = self.init_hidden(x.size(0))
            outputs = self(x, h0)
            y_hat = F.softmax(outputs, dim=1)
            y_hat = y_hat.data.numpy() if not self.use_cuda else y_hat.cpu().data.numpy()
            y_pred.append(y_hat)
            # empty cuda cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred)
        return y_pred[:, pos_label] if binary else y_pred

    def predict(self, data_loader, binary=True, pos_label=1, threshold=0.5, return_proba=False, topSelection=None):
        """
        If binary classification, use threshold on positive class
        If multinomial, just select the max probability as the predicted class
        :param data_loader:
        :param binary:
        :param pos_label:
        :param threshold:
        :return:
        """
        proba = self.predict_proba(data_loader, binary, pos_label)
        if topSelection is not None and topSelection < proba.shape[0]:
            threshold = proba[np.argsort(proba)[-topSelection - 1]]
        if binary:
            pred = np.array([1 if p > threshold else 0 for p in proba])
        else:
            pred = np.argmax(proba, 1)

        if return_proba:
            return (proba, pred)
        else:
            return pred



class Dense4012FrameRNN(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(Dense4012FrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "Dense4012FrameRNN"
        input_shape = kwargs.get("input_shape", (8, 64, 64))

        seq_output_size = kwargs.get("seq_output_size", 128)
        seq_dropout = kwargs.get("seq_dropout", 0.1)
        seq_attention = kwargs.get("seq_attention", True)
        seq_bidirectional = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len = kwargs.get("seq_max_seq_len", 13)
        seq_rnn_type = kwargs.get("rnn_type", "LSTM")
        pretrained = kwargs.get("pretrained", True)
        requires_grad = kwargs.get("requires_grad", True)

        self.fenc, _ = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size = self.get_frm_output_size(input_shape)
        # print(input_shape)
        # print(frm_output_size)

        self.senc = RNN(n_classes=n_classes, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0, 1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        # print(x.shape)
        frm_output_size = self.fenc.forward(x).view(-1).size()[0]
        return frm_output_size

class Dense4012FrameRNN_3slices(MRISequenceNet):
    def __init__(self, densenet_ckpt, n_classes, use_cuda, **kwargs):
        super(Dense4012FrameRNN_3slices, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "Dense4012FrameRNN"
        input_shape = kwargs.get("input_shape", (3, 64, 64))

        seq_output_size = kwargs.get("seq_output_size", 128)
        seq_dropout = kwargs.get("seq_dropout", 0.1)
        seq_attention = kwargs.get("seq_attention", True)
        seq_bidirectional = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len = kwargs.get("seq_max_seq_len", 13)
        seq_rnn_type = kwargs.get("rnn_type", "LSTM")
        pretrained = kwargs.get("pretrained", True)
        requires_grad = kwargs.get("requires_grad", True)

        self.fenc, _ = densenet_40_12_bc_3slices(densenet_ckpt, pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size = self.get_frm_output_size(input_shape)
        # print(input_shape)
        # print(frm_output_size)

        self.senc = RNN(n_classes=n_classes, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0, 1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size = self.fenc.forward(x).view(-1).size()[0]
        return frm_output_size

        
class Three_plus_three_cnn_lstm(nn.Module):

    """
    Args:
        num_labels (int): Number of output classes (e.g., for classification tasks on expert derived features).
        embed_dim (int, optional): Dimension of the input ECG feature embeddings. Default: 768.
        hidden_dim (int, optional): Dimension of the hidden layer in MLP. Default: 256.
        dropout_rate (float, optional): Dropout probability to prevent overfitting. Default: 0.2.
        label_weights (bool, optional): Whether to apply label weighting in the loss function. Default: False.
    """
    def __init__(self, densenet_ckpt):
        super(Three_plus_three_cnn_lstm, self).__init__()

        self.mri_encoder1 = Dense4012FrameRNN_3slices(densenet_ckpt, 2, True)
        self.mri_encoder2 = Dense4012FrameRNN_3slices(densenet_ckpt, 2, True)
        self.classifier = nn.Linear(4, 2)
    

    def forward(self, mris):

        mris1 = mris[:, :, 0:3, :, :]  
        mris2 = mris[:, :, 3:6, :, :]  

        h1 = self.mri_encoder1.init_hidden(mris1.size(0))
        mri_features1 = self.mri_encoder1(mris1, h1)

        h2 = self.mri_encoder2.init_hidden(mris2.size(0))
        mri_features2 = self.mri_encoder2(mris2, h2)

        mri_features = torch.cat([mri_features1, mri_features2], dim=1)  
        mri_features = self.classifier(mri_features)

        return mri_features
