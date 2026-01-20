# STaR-DR — Sample-Efficient Drug Response Prediction under Domain Shift

**STaR-DR (Staged Transfer of Representations for Drug Response)** is a deep learning framework designed to improve drug-response prediction (DRP) transfer from preclinical cancer cell lines to patient tumors under strong biological domain shift.

The core idea is to explicitly separate representation learning from task supervision, enabling few-shot adaptation to patient data with minimal labeled samples.
This repo provides the training and evaluation scripts used to study sample-efficient adaptation to TCGA.

## Key Idea
Predictive models trained on pharmacogenomic cell-line screens (source domain) often fail to generalize to patient tumors (target domain).  
STaR-DR aims to reduce the amount of labeled clinical supervision needed by learning transferable latent representations.

STaR-DR follows a 3-stage training protocol:

1. **(P1) Unsupervised pretraining**  
   Independent autoencoders learn *cell* and *drug* latent representations from large collections of unlabeled pharmacogenomic data.

2. **(P2) Supervised alignment on cell lines**  
   Pretrained encoders are jointly fine-tuned with a lightweight MLP classifier on labeled cell–drug response pairs.

3. **(P3) Few-shot adaptation to patients**  
   The model is adapted to TCGA tumors using very limited labeled samples (few-shot).

## Datasets
Experiments cover increasing levels of domain shift using:
- **CTRP + GDSC** (training source dataset)
- **CCLE** (cross-dataset evaluation)
- **TCGA** (patient-level transfer / few-shot adaptation)

**Modalities**:
- **Cells:** gene expression (+ mutation when available)
- **Drugs:** molecular descriptors + Morgan fingerprints  
TCGA experiments use gene expression only.

The datasets used in this project were obtained from the official [DeepDRA repository](https://git.dml.ir/taha.mohammadzadeh/DeepDRA).

## Baseline
A matched **single-phase AE-MLP baseline** (DeepDRA-derived) is implemented for comparison.
It uses the same architecture but is trained end-to-end on labeled data without unsupervised pretraining.

## Disclaimer
This repository is for research and educational purposes only and is not intended for clinical use.

## Contact
Camille Jimenez Cortes — [@cjimenezcortes](https://github.com/cjimenezcortes)
