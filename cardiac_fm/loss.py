import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature).log())

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, mri_features, ecg_features):
        logits_per_mri = mri_features @ ecg_features.T / self.temperature.exp()
        logits_per_ecg = ecg_features @ mri_features.T / self.temperature.exp()
        return logits_per_mri, logits_per_ecg

    def forward(self, mri_features, ecg_features):
        device = mri_features.device
        logits_per_mri, logits_per_ecg = self.get_logits(mri_features, ecg_features)

        labels = self.get_ground_truth(device, logits_per_mri.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_mri, labels) +
            F.cross_entropy(logits_per_ecg, labels)
        ) / 2
        
        return total_loss
 

