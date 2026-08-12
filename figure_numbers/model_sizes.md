# Model sizes — verified parameter counts (for R3Q6, foundation-model definition)

Parameter counts measured by loading each released checkpoint's model weights and summing `numel`
(optimizer state excluded), or by instantiating the model where the checkpoint is a frozen TorchScript.
CineMA is measured from its released size config (see note). Measured on 2026-08 with `mri_env`.

| model | params | modality | how measured | source file |
|---|---|---|---|---|
| **CARDIAC-FM (total)** | **~183 M** | ECG + MRI | sum of the two towers | — |
| — ECG-FM tower | 90.9 M | ECG | `build_model_from_checkpoint` → sum numel | `weights/ecgfm_mimic_iv_physionet.pt` |
| — MRI encoder | 92.1 M (enc) / 118.5 M (with decoder) | MRI | instantiated CineMAE (base cfg), sum numel | `common/mri_encoder/cinema_mae.py` |
| ECGFounder | 30.9 M | ECG | checkpoint `state_dict`, sum numel | `ECGFounder_DeepSSL/ECGFounder/checkpoint/12_lead_ECGFounder.pth` (353 MB, has optimizer) |
| DeepECG-SSL | 90.9 M | ECG | checkpoint `model` key, sum numel | `ECGFounder_DeepSSL/deepecg/SSL_pretrained.pt` (1.1 GB, has optimizer) |
| DeepECG-SL | ~1.5 M | ECG | frozen TorchScript; confirmed by 6.9 MB file (~4 B/param) | `ECGFounder_DeepSSL/deepecg/efficientnet_77.pt` |
| CineMA (base) | ~85–92 M | cardiac MRI | from config (see note) | `CineMA_ref/cinema/vit.py` size dict |
| CineMA (large) | ~300 M | cardiac MRI | from config | " |
| CineMA (huge) | ~630 M | cardiac MRI | from config | " |

## Key facts for R3Q6
- **CARDIAC-FM (~183 M) is the largest of every ECG model benchmarked** (ECG-FM 91 M, DeepECG-SSL 91 M,
  ECGFounder 31 M, DeepECG-SL 1.5 M). Directly refutes "the parameter count is small."
- **The MRI tower equals CineMA-base** (768/12/12, dec 512/8/16 — byte-identical config), a recognized
  cardiac-imaging foundation model, and **both pretrain on UK Biobank MRI** (same data source). So the
  MRI-side scale and data are on par with an established cardiac FM.
- ECGFounder's "large-scale" descriptor refers to its **pretraining data** (~millions of ECGs), not its
  parameter count (31 M).

## Provenance caveats
- **CineMA size is from its released config**, not a direct instantiation (its code needs `timm`, not in
  `mri_env`; we did not install on the login node). But CineMA-`base` config is byte-identical to our MRI
  encoder, which measured 92.1 M, so ~92 M is reliable. CineMA uses 4 views (SAX + 2/3/4-CH) vs our 3, a
  ~1–3 M conv-stem difference. For an exact released-checkpoint number, count `mathpluscode/CineMA` on a
  compute node.
- **DeepECG-SL** is a frozen TorchScript (`.parameters()` alone undercounts); the 6.9 MB file size confirms
  ~1.5 M params. It is a compact EfficientNet-V2 (12×2500 → 77 classes), genuinely small.
- ECG-FM / MRI totals: 90.9 + 92.1 ≈ 183 M (the "~187 M" in drafts is this ± projection heads; quote ~183 M).
