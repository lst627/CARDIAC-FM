"""
Self-contained ECG-only models + risk fusion + cosine schedule for the CHS/MESA
external validation. The ECG-side definitions are kept independent of the MRI encoder
so ECG-only tasks import only the dependencies they use and run cleanly in mri_env.
"""
import torch
import torch.nn as nn
import numpy as np
from fairseq_signals.models import build_model_from_checkpoint


# ------------------------- models (verbatim from model.py) -------------------------
class ECGFM(nn.Module):
    def __init__(self, ecgfm_ckpt, num_labels=1, embed_dim=768, hidden_dim=512,
                 dropout_rate=0.2, use_label_weights=False):
        super(ECGFM, self).__init__()
        self.ecg_encoder = build_model_from_checkpoint(ecgfm_ckpt)
        self.pred = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, ecgs):
        ecg_features = self.ecg_encoder.extract_features(
            source=ecgs["net_input"]["source"], padding_mask=ecgs["net_input"]["padding_mask"])
        x = ecg_features["x"]
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        x = self.pred(x)
        return x


class CARDIACFM_ECG(nn.Module):
    def __init__(self, ecgfm_ckpt, training_type="full", num_labels=1, embed_dim=768,
                 hidden_dim=512, dropout_rate=0.2, use_label_weights=False,
                 cardiacfm_pretrained_ckpt=None):
        super(CARDIACFM_ECG, self).__init__()
        self.ecg_encoder_multi = build_model_from_checkpoint(ecgfm_ckpt)
        self.ecg_projection_multi = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, hidden_dim),
        )
        if cardiacfm_pretrained_ckpt is not None:
            with open(cardiacfm_pretrained_ckpt, "rb") as f:
                state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
            # our stage-1 CL saver wraps the weights in a training-state dict; unwrap to the
            # bare state_dict that the CARDIAC-FM loader below expects.
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            state_projection = {k.replace("ecg_projection.", ""): v
                                for k, v in state.items() if "ecg_projection." in k}
            state = {k.replace("ecg_encoder.", ""): v
                     for k, v in state.items() if "ecg_encoder." in k}
            self.ecg_encoder_multi.load_state_dict(state, strict=True)
            self.ecg_projection_multi.load_state_dict(state_projection, strict=True)

        self.pred = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, ecgs):
        ecg_features = self.ecg_encoder_multi.extract_features(
            source=ecgs["net_input"]["source"], padding_mask=ecgs["net_input"]["padding_mask"])
        x = ecg_features["x"]
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        x = self.ecg_projection_multi(x)
        x = self.pred(x)
        return x


# ------------------------- cosine schedule (verbatim from utils.py) -------------------------
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


# ------------------------- risk fusion (verbatim from model.py) -------------------------
def CHARGE_AF(df):
    charge_age    = df["age"] / 5 * 0.508
    charge_race   = (df["race"] == 1).astype(float) * 0.465
    charge_height = df["ht"] / 10 * 0.248
    charge_weight = df["wt"] / 15 * 0.115
    charge_sbp    = df["sbp"] / 20 * 0.197
    charge_dbp    = df["dbp"] / 10 * (-0.101)
    charge_smk    = (df["cursmoke"] == 1).astype(float) * 0.359
    charge_htnmed = (df["htnmed"] == 1).astype(float) * 0.349
    charge_prevdm = (df["prevdm"] == 1).astype(float) * 0.237
    charge_prevhf = (df["prevhf"] == 1).astype(float) * 0.701
    charge_prevmi = (df["prevmi"] == 1).astype(float) * 0.496
    charge_sum = (charge_age + charge_race + charge_height + charge_weight + charge_sbp
                  + charge_dbp + charge_smk + charge_htnmed + charge_prevdm
                  + charge_prevhf + charge_prevmi)
    return 1 - (0.9718412736 ** (charge_sum - 12.5815600))


def PREVENT_HF(df):
    sex_male = df["sex"] == 1
    age10 = (df["age"] - 55.0) / 10.0
    sbp_low  = (np.minimum(df["sbp"], 110.0) - 110.0) / 20.0
    sbp_high = (np.maximum(df["sbp"], 110.0) - 130.0) / 20.0
    bmi_low  = (np.minimum(df["bmi"], 30.0) - 25.0) / 5.0
    bmi_high = (np.maximum(df["bmi"], 30.0) - 30.0) / 5.0
    egfr_low  = (np.minimum(df["egfr"], 60.0) - 60.0) / (-15.0)
    egfr_high = (np.maximum(df["egfr"], 60.0) - 90.0) / (-15.0)
    prevdm_i   = (df["prevdm"] == 1).astype(float)
    cursmoke_i = (df["cursmoke"] == 1).astype(float)
    htnmed_i   = (df["htnmed"] == 1).astype(float)
    logodds_w = (-4.310409 + 0.8998235*age10 - 0.4559771*sbp_low + 0.3576505*sbp_high
                 + 1.038346*prevdm_i + 0.583916*cursmoke_i - 0.0072294*bmi_low
                 + 0.2997706*bmi_high + 0.7451638*egfr_low + 0.0557087*egfr_high
                 + 0.3534442*htnmed_i - 0.0981511*htnmed_i*sbp_high - 0.0946663*age10*sbp_high
                 - 0.3581041*age10*prevdm_i - 0.1159453*age10*cursmoke_i
                 - 0.0038780*age10*bmi_high - 0.1884289*age10*egfr_low)
    logodds_m = (-3.946391 + 0.8972642*age10 - 0.6811466*sbp_low + 0.3634461*sbp_high
                 + 0.923776*prevdm_i + 0.5023736*cursmoke_i - 0.0485841*bmi_low
                 + 0.3726929*bmi_high + 0.6926917*egfr_low + 0.0251827*egfr_high
                 + 0.2980922*htnmed_i - 0.0497731*htnmed_i*sbp_high - 0.1289201*age10*sbp_high
                 - 0.3040924*age10*prevdm_i - 0.1401688*age10*cursmoke_i
                 + 0.0068126*age10*bmi_high - 0.1797778*age10*egfr_low)
    logodds = np.where(sex_male, logodds_m, logodds_w)
    return 1 / (1 + np.exp(-logodds))


def CARDIACFM_ECG_Risk_AF(df, seed):
    if seed not in (1, 2, 3, 4):
        raise ValueError("seed must be one of 1, 2, 3, 4")
    coefs = {1: (-4.057, 8.468, 27.948), 2: (-3.715, 12.537, 34.139),
             3: (-4.049, 5.555, 27.302), 4: (-3.727, 6.621, 31.046)}
    intercept, beta_m, beta_r = coefs[seed]
    return intercept + beta_m * df["model_score"].values + beta_r * df["risk_score"].values


def CARDIACFM_ECG_Risk_HF(df, seed):
    if seed not in (1, 2, 3, 4):
        raise ValueError("seed must be one of 1, 2, 3, 4")
    coefs = {1: (-5.477, 11.142, 12.457), 2: (-5.621, 21.670, 16.661),
             3: (-5.632, 13.969, 17.031), 4: (-5.820, 15.740, 16.020)}
    intercept, beta_m, beta_r = coefs[seed]
    return intercept + beta_m * df["model_score"].values + beta_r * df["risk_score"].values
