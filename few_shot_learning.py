# =============================
# Few-shot learning on TCGA
# =============================
"""
# This script implements zero-shot and few-shot transfer learning on TCGA
# for drug response prediction, using pretrained AE-MLP or STaR-DR(P1,P2)
#
# Two evaluation regimes are supported:
# 1) Zero-shot (K = 0):
#    - Direct evaluation on TCGA without any adaptation.
#    - TCGA inputs are normalized using source train-time L2 norms
#
# 2) Few-shot (K > 0):
#    - A K-support set is sampled from TCGA using stratified splitting.
#    - RandomUnderSampler (RUS) is applied on the support set to mitigate class imbalance.
#    - Model adaptation is performed in two phases:
#
#      Phase A (Warm-up):
#        - Train lightweight linear adapters on latent representations
#          and the MLP classifier only.
#        - Encoders remain fully frozen.
#
#      Phase B (Partial unfreezing):
#        - Unfreeze only the last linear layer of the cell encoder
#          (and optionally the drug encoder).
#        - Continue training adapters + MLP.
#        - Apply L2-SP regularization to constrain parameters
#          close to their pretrained values.
#
# For each K, multiple runs with different random seeds are executed.
# Performance is aggregated as mean ± standard deviation across runs,
# and reported using AUC and AUPRC metrics.
#
# Outputs:
# - CSV file with aggregated results per K
# - Performance plot vs effective support size (post-RUS)

"""
import copy
import os
import numpy as np
import pandas as pd
import torch
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler

from mlp import MLP
from utils import DATA_MODALITIES, TCGA_DATA_FOLDER, TCGA_SCREENING_DATA
from data_loader import RawDataLoader
from evaluation import Evaluation

# ====== Device & Seeds ======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def set_seed(seed=42):
    """
    Set random seeds for Python, NumPy and PyTorch to ensure reproducibility.
    Parameters:
    seed (int): default=42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====== Utils (grad control, last layer, L2-SP) ======
def set_requires_grad(module, flag: bool):
    """
    Enable or disable gradient computation for all parameters of a module.
    Useful to freeze/unfreeze blocks during few-shot adaptation.

    Parameters:

    module (torch.nn.Module): The module to (un)freeze.
    flag (bool): If True, parameters will be trainable; if False, frozen.
    """
    for p in module.parameters():
        p.requires_grad = flag

def get_last_linear(module: nn.Module):
    """
    Retrieve the last nn.Linear layer found in a module.
    This is used to implement partial unfreezing (only the last layer
    of an encoder is trained during few-shot adaptation).

    Parameters:
    module (torch.nn.Module): PyTorch module

    Returns:
    nn.Linear or None: The last linear layer if found, otherwise None.
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last

def snapshot_state_dict(modules):
    """
    Snapshot state_dicts of a list of modules for L2-SP regularization.

    Parameters:
    modules (list[torch.nn.Module]): Modules to snapshot.

    Return a dict {module_id: state_dict} as reference for L2-SP.
    """
    return {id(m): {k: v.detach().clone() for k, v in m.state_dict().items()} for m in modules}

def l2sp_loss(modules, refs, lam=1e-4):
    """
    Compute L2-SP regularization loss for a set of modules.

    L2-SP penalizes deviation from a pretrained reference:
    sum ||theta - theta_pretrained||^2 over trainable parameters.

    Parameters:
    - modules (list[torch.nn.Module]): Modules to regularize.
    - refs (dict): Reference snapshots returned by snapshot_state_dict.
    - lam (float): default=1e-4; Regularization strength. If lam <= 0, returns 0.

    Return torch.Tensor : Scalar regularization loss on the current device.
    """
    if lam <= 0:
        return torch.tensor(0.0, device=device)
    reg = torch.tensor(0.0, device=device)
    for m in modules:
        ref_sd = refs.get(id(m), None)
        if ref_sd is None:
            continue
        for (n, p) in m.named_parameters():
            if not p.requires_grad:
                continue
            ref_w = ref_sd.get(n, None)
            if ref_w is None:
                continue
            reg = reg + torch.sum((p - ref_w.to(p.device)) ** 2)
    return lam * reg

# ====== Data helpers ======
def make_zero_shot_loader(df_cell, df_drug, y_np, *, batch_size=64, cell_norms, drug_norms):
    """
    Build a DataLoader for zero-shot evaluation (on TCGA).

    The key point is to normalize TCGA inputs using the *source training norms*
    (computed on the source dataset during AE-MLP training) to avoid leakage
    and keep inputs in the same scale as the pretrained encoders.

    Parameters:
    - df_cell (pd.DataFrame): Cell modality features for TCGA pairs.
    - df_drug (pd.DataFrame): Drug modality features for TCGA pairs.
    - y_np (array-like): Binary labels (0/1) for the TCGA pairs.
    - batch_size (int): default=64; Batch size for the DataLoader.
    - cell_norms (torch.Tensor): Train-time L2 norms for the cell features (source dataset).
    - drug_norms (torch.Tensor): Train-time L2 norms for the drug features (source dataset).

    Return : torch.utils.data.DataLoader: DataLoader yielding (X_cell, X_drug, y) batches.
    """
    x_cell = torch.tensor(df_cell.values, dtype=torch.float32)
    x_drug = torch.tensor(df_drug.values, dtype=torch.float32)
    y_t    = torch.tensor(np.asarray(y_np).reshape(-1, 1), dtype=torch.float32)

    # Use provided train-time norms with stability guard
    thr = 1e-6
    cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
    drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)

    # normalisation with source norm
    x_cell = x_cell / cell_norms
    x_drug = x_drug / drug_norms
    return DataLoader(TensorDataset(x_cell, x_drug, y_t), batch_size=batch_size, shuffle=False)

def make_few_shot_loaders(
    df_cell, 
    df_drug, 
    y_np, 
    K, 
    cell_norms, 
    drug_norms, 
    seed=42, 
    batch_size=64, 
):
    """
    Build few-shot train/validation DataLoaders from TCGA using a K-support split.

    The split is stratified (StratifiedShuffleSplit) to preserve class balance
    in the support set. RandomUnderSampler (RUS) is applied on the
    support set only, to reduce majority-class dominance.

    Normalization uses the *source train-time norms* to maintain consistency
    with the pretrained encoders.

    Parameters:
    - df_cell (pd.DataFrame): Cell features for TCGA pairs.
    - df_drug (pd.DataFrame): Drug features for TCGA pairs.
    - y_np (array-like): Binary labels (0/1).
    - K (int): Support set size before RUS.
    - cell_norms (torch.Tensor): Source train-time L2 norms for cell features.
    - drug_norms (torch.Tensor): Source train-time L2 norms for drug features.
    - seed (int); default=42: Random seed controlling the split and RUS.
    - batch_size (int); default=64; Batch size for DataLoaders.

    Return:
    - train_loader : DataLoader (few-shot support loader).
    - val_loader : DataLoader (validation loader on the remaining TCGA samples).
    - info (dict): Post-RUS metadata: K_post, n_pos_post, n_neg_post.
    """
    full_df = pd.concat([df_cell, df_drug], axis=1)
    n_cell  = df_cell.shape[1]
    y_np    = np.asarray(y_np).ravel().astype(int)
    sss     = StratifiedShuffleSplit(n_splits=1, train_size=K, random_state=seed)
    idx_tr, idx_val = next(sss.split(full_df, y_np))

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=seed)
    X_train_bal, y_train_bal = rus.fit_resample(full_df.iloc[idx_tr], y_np[idx_tr])

    # Split back into modalities
    x_cell_tr = X_train_bal.iloc[:, :n_cell].values
    x_drug_tr = X_train_bal.iloc[:,  n_cell:].values
    y_tr      = y_train_bal.astype(float).reshape(-1, 1)

    x_cell_v  = full_df.iloc[idx_val, :n_cell].values
    x_drug_v  = full_df.iloc[idx_val,  n_cell:].values
    y_v       = y_np[idx_val].astype(float).reshape(-1, 1)

    # Tensors + normalization
    thr = 1e-6
    cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
    drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)
    x_cell_tr_t = torch.tensor(x_cell_tr, dtype=torch.float32)
    x_drug_tr_t = torch.tensor(x_drug_tr, dtype=torch.float32)
    x_cell_v_t  = torch.tensor(x_cell_v,  dtype=torch.float32)
    x_drug_v_t  = torch.tensor(x_drug_v,  dtype=torch.float32)
    y_tr_t      = torch.tensor(y_tr,      dtype=torch.float32)
    y_v_t       = torch.tensor(y_v,       dtype=torch.float32)

    x_cell_tr_t = x_cell_tr_t / cell_norms
    x_drug_tr_t = x_drug_tr_t / drug_norms
    x_cell_v_t  = x_cell_v_t  / cell_norms
    x_drug_v_t  = x_drug_v_t  / drug_norms

    train_loader = DataLoader(TensorDataset(x_cell_tr_t, x_drug_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader   = DataLoader(TensorDataset(x_cell_v_t,  x_drug_v_t,  y_v_t),
                              batch_size=batch_size, shuffle=False)
    
    # Post-RUS info (for logging/plotting)
    K_post = len(train_loader.dataset)
    y_tr_t_all = train_loader.dataset.tensors[2]
    n_pos_post = int((y_tr_t_all > 0.5).sum().item())
    n_neg_post = int(K_post - n_pos_post)
    info = {"K_post": K_post, "n_pos_post": n_pos_post, "n_neg_post": n_neg_post}
    
    return train_loader, val_loader, info

# ====== Eval helper ======
def forward_predict(enc_c, enc_d, adapters, mlp_mod, loader, device, threshold=0.5):
    """
    Run inference on a loader and compute evaluation metrics.

    This helper encodes cell/drug inputs, optionally applies linear adapters,
    concatenates latent representations, and predicts with the MLP head.

    Parameters:
    - enc_c (torch.nn.Module): Cell encoder.
    - enc_d (torch.nn.Module): Drug encoder.
    - adapters (tuple(nn.Module, nn.Module) or None)
        (adapter_cell, adapter_drug) applied on z_cell/z_drug if provided.
    - mlp_mod (torch.nn.Module): MLP classifier head.
    - loader : DataLoader: Loader yielding (X_cell, X_drug, y).
    - device (torch.device): Device used for computation.
    - threshold (float): default=0.5; Decision threshold for binarizing probabilities.

    Return (dict): Evaluation metrics returned by Evaluation.evaluate.
    """

    enc_c.eval(); enc_d.eval(); mlp_mod.eval()
    if adapters is not None:
        ad_c, ad_d = adapters
        ad_c.eval(); ad_d.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for Xc, Xd, yt in loader:
            Xc, Xd, yt = Xc.to(device), Xd.to(device), yt.to(device)
            zc = enc_c(Xc); zd = enc_d(Xd)
            if adapters is not None:
                zc = ad_c(zc); zd = ad_d(zd)
            y  = mlp_mod(torch.cat([zc, zd], dim=1))
            y_true_all.append(yt.cpu()); y_pred_all.append(y.cpu())
    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)
    return Evaluation.evaluate(y_true, y_pred, show_plot=False, threshold=threshold, verbose=False)

# ====== One few-shot run (K support) ======
def run_one_few_shot(
    df_cell, df_drug, y_np, K_support,
    encoder_cell, encoder_drug, mlp, *,
    cell_norms, drug_norms, epochs=25, batch_size=64,
    lr_ae=1e-5, lr_mlp=3e-4,
    seed=42,
    warmup_epochs=5,                 # adapters + MLP only
    freeze_drug_encoder=True, # keep drug encoder frozen when partially unfreezing
    l2sp_lambda=1e-4,
    run_id=0
):
    """
    Perform one few-shot adaptation run on TCGA for a given support size K.

    Two modes:
    - Zero-shot (K_support == 0): directly evaluate pretrained encoders + MLP
      on TCGA using source train-time normalization.
    - Few-shot (K_support > 0): build a K-support set, apply RUS,
      then adapt the model using:
        Phase A: warm-up training of adapters + MLP only
        Phase B: partial unfreezing (last linear layer of encoders) + adapters + MLP
      with L2-SP regularization to stay close to pretrained weights.

    Parameters:
    - df_cell, df_drug (pd.DataFrame): TCGA input features (aligned to source feature space).
    - y_np : (array-like): TCGA binary labels.
    - K_support (int): Support set size (before RUS). If 0, runs zero-shot.
    - encoder_cell, encoder_drug (torch.nn.Module): Pretrained encoders from the source dataset.
    - mlp (torch.nn.Module): Pretrained MLP head from the source dataset.
    - cell_norms, drug_norms (torch.Tensor): Source train-time feature norms used for normalization.
    - epochs (int); default=25; Number of adaptation epochs in Phase B.
    - batch_size (int); default=64; Batch size for loaders.
    - lr_ae (float); default=1e-5; Learning rate for partially unfrozen encoder layers.
    - lr_mlp (float; default=3e-4; Learning rate for adapters + MLP head.
    - seed (int); default=42; Seed controlling split and optimization reproducibility.
    - warmup_epochs (int); default=5; Number of warm-up epochs (adapters + MLP only).
    - freeze_drug_encoder (bool); default=True; If True, keep drug encoder frozen during partial unfreezing.
    - l2sp_lambda (float); default=1e-4; L2-SP regularization strength.
    - run_id (int); default=0; Identifier used for tracking runs (logging/saving if needed).

    Return (dict): Evaluation metrics (AUC/AUPRC) plus metadata (K_post, n_pos_post, n_neg_post).
    """
    set_seed(seed)

    # ---------- Zero-shot ----------
    if K_support == 0:

        val_loader = make_zero_shot_loader(df_cell, df_drug, y_np, batch_size=batch_size,
                                           cell_norms=cell_norms, drug_norms=drug_norms)
        enc_c_ = copy.deepcopy(encoder_cell).to(device).float()
        enc_d_ = copy.deepcopy(encoder_drug).to(device).float()
        mlp_   = copy.deepcopy(mlp).to(device).float()
        res = forward_predict(enc_c_, enc_d_, None, mlp_, val_loader, device)
        res["K_post"] = 0
        res["n_pos_post"] = 0
        res["n_neg_post"] = 0
        return res

    
    # ---------- Few-shot ----------

    train_loader, val_loader, info = make_few_shot_loaders(
            df_cell, df_drug, y_np, K_support,
            cell_norms=cell_norms, drug_norms=drug_norms,
            seed=seed, batch_size=batch_size
    )
        

    # Clone modules
    enc_c_ = copy.deepcopy(encoder_cell).to(device).float()
    enc_d_ = copy.deepcopy(encoder_drug).to(device).float()
    mlp_   = copy.deepcopy(mlp).to(device).float()
        
    # L2-SP refs
    ref_sd = snapshot_state_dict([enc_c_, enc_d_, mlp_])
        
    # Adapters (identity init) — dimensions from a mini-forward
    with torch.no_grad():
        Xc0, Xd0, _ = next(iter(train_loader))
        zc_dim = enc_c_(Xc0[:1].to(device)).shape[1]
        zd_dim = enc_d_(Xd0[:1].to(device)).shape[1]
    adapter_cell = nn.Linear(zc_dim, zc_dim, bias=True).to(device)
    adapter_drug = nn.Linear(zd_dim, zd_dim, bias=True).to(device)
    with torch.no_grad():
        nn.init.eye_(adapter_cell.weight); adapter_cell.bias.zero_()
        nn.init.eye_(adapter_drug.weight); adapter_drug.bias.zero_()

    # crit & weighting
    bce = nn.BCELoss()

    def eval_on_val():
        """
        Run a full evaluation pass on the validation loader.

        Uses the current modules:
        - enc_c_ / enc_d_ (encoders)
        - adapter_cell / adapter_drug (latent adapters)
        - mlp_ (classification head)

        Return (float, float): (AUPRC, AUC) computed on the full validation set.
        """
        y_true_all, y_score_all = [], []
        enc_c_.eval(); enc_d_.eval(); adapter_cell.eval(); adapter_drug.eval(); mlp_.eval()
        with torch.no_grad():
            for Xc, Xd, yt in val_loader:
                Xc, Xd, yt = Xc.to(device), Xd.to(device), yt.to(device)
                
                # Encode → adapt → classify
                zc = adapter_cell(enc_c_(Xc)); zd = adapter_drug(enc_d_(Xd))
                y  = mlp_(torch.cat([zc, zd], dim=1))
                y_true_all.append(yt.cpu()); y_score_all.append(y.cpu())
                
        # Concatenate all batches into single tensors
        y_true = torch.cat(y_true_all, dim=0)
        y_pred = torch.cat(y_score_all, dim=0)
        
        # Silent evaluation (verbose=False) to avoid flooding logs
        res = Evaluation.evaluate(y_true, y_pred, show_plot=False, verbose=False)
        return float(res['AUPRC']), float(res['AUC'])

    # Checkpoint helper: keep best model according to validation AUPRC
    def snapshot():
        """
        Save a deep copy of the current trainable state.

        Store:
        - both encoders
        - both adapters
        - the MLP head

        Return (dict): Dictionary of state_dict copies for later restoration.
        """    
        return {
            'enc_c': copy.deepcopy(enc_c_.state_dict()),
            'enc_d': copy.deepcopy(enc_d_.state_dict()),
            'ad_c':  copy.deepcopy(adapter_cell.state_dict()),
            'ad_d':  copy.deepcopy(adapter_drug.state_dict()),
            'mlp':   copy.deepcopy(mlp_.state_dict())
        }

    # Baseline before any adaptation (used to initialize "best" checkpoint)
    auprc0, _ = eval_on_val()
    best_auprc = auprc0
    best_ckpt  = snapshot()

    # ==============================================================
    # Phase A: Warm-up (train adapters + MLP only)
    # ==============================================================
    # Goal: adapt a lightweight head to TCGA while keeping encoders frozen.
    set_requires_grad(enc_c_, False); set_requires_grad(enc_d_, False)
    set_requires_grad(adapter_cell, True); set_requires_grad(adapter_drug, True)
    set_requires_grad(mlp_, True)

    params_warm = list(adapter_cell.parameters()) + list(adapter_drug.parameters()) + list(mlp_.parameters())
    opt_warm = optim.Adam([{'params': params_warm, 'lr': lr_mlp, 'weight_decay': 1e-5}])

    for _ in range(max(0, warmup_epochs)):
        # encoders stay in eval mode (frozen); adapters/MLP are trained
        enc_c_.eval(); enc_d_.eval(); adapter_cell.train(); adapter_drug.train(); mlp_.train()
        for Xc, Xd, yt in train_loader:
            Xc, Xd, yt = Xc.to(device), Xd.to(device), yt.to(device)
            opt_warm.zero_grad()
            zc = adapter_cell(enc_c_(Xc)); zd = adapter_drug(enc_d_(Xd))
            y  = mlp_(torch.cat([zc, zd], dim=1))
            loss = bce(y.float(), yt.float())
            loss = loss + l2sp_loss([mlp_], ref_sd, l2sp_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_warm, 1.0)
            opt_warm.step()
            
        # Track best checkpoint by validation AUPRC
        auprc, _ = eval_on_val()
        if auprc > best_auprc:
            best_auprc = auprc
            best_ckpt = snapshot()

    # ==============================================================
    # Phase B: Partial unfreeze encoders (last linear layer only)
    # ==============================================================
    # Goal: allow limited encoder adaptation while preventing overfitting.
    set_requires_grad(enc_c_, False); set_requires_grad(enc_d_, False)
    last_c = get_last_linear(enc_c_); last_d = get_last_linear(enc_d_)

    # Unfreeze only the last Linear of the cell encoder
    if last_c is not None:
        last_c.weight.requires_grad = True
        if last_c.bias is not None: last_c.bias.requires_grad = True

    # Optionally unfreeze last layer of drug encoder (by default: kept frozen)
    if (not freeze_drug_encoder) and (last_d is not None):
        last_d.weight.requires_grad = True
        if last_d.bias is not None: last_d.bias.requires_grad = True
    set_requires_grad(adapter_cell, True); set_requires_grad(adapter_drug, True); set_requires_grad(mlp_, True)

    # Separate parameter groups for different learning rates
    params_enc  = [p for p in enc_c_.parameters() if p.requires_grad]
    if not freeze_drug_encoder:
        params_enc += [p for p in enc_d_.parameters() if p.requires_grad]
    params_head = list(adapter_cell.parameters()) + list(adapter_drug.parameters()) + list(mlp_.parameters())

    opt  = optim.Adam([{'params': params_enc,  'lr': lr_ae,  'weight_decay': 1e-5},
                       {'params': params_head, 'lr': lr_mlp, 'weight_decay': 1e-5}])
    sched = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=5, verbose=False)

    for _ in range(epochs):
        adapter_cell.train(); adapter_drug.train(); mlp_.train()
        if last_c is not None: last_c.train()
        if (not freeze_drug_encoder) and (last_d is not None): last_d.train()

        for Xc, Xd, yt in train_loader:
            Xc, Xd, yt = Xc.to(device), Xd.to(device), yt.to(device)
            opt.zero_grad()
            zc = adapter_cell(enc_c_(Xc)); zd = adapter_drug(enc_d_(Xd))
            y  = mlp_(torch.cat([zc, zd], dim=1))
            loss = bce(y.float(), yt.float())
            mods = [mlp_, adapter_cell, adapter_drug]
            if any(p.requires_grad for p in enc_c_.parameters()): mods.append(enc_c_)
            if any(p.requires_grad for p in enc_d_.parameters()): mods.append(enc_d_)
            loss = loss + l2sp_loss(mods, ref_sd, l2sp_lambda)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_enc + params_head, 1.0)
            opt.step()

        # validation
        enc_c_.eval(); enc_d_.eval(); adapter_cell.eval(); adapter_drug.eval(); mlp_.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xc, Xd, yt in val_loader:
                Xc, Xd, yt = Xc.to(device), Xd.to(device), yt.to(device)
                zc = adapter_cell(enc_c_(Xc)); zd = adapter_drug(enc_d_(Xd))
                y  = mlp_(torch.cat([zc, zd], dim=1))
                val_loss += nn.BCELoss()(y.float(), yt.float()).item()
        val_loss /= max(1, len(val_loader))
        sched.step(val_loss)

        auprc, _ = eval_on_val()
        if auprc > best_auprc:
            best_auprc = auprc
            best_ckpt = snapshot()

    # Restore best checkpoint and evaluate once 
    enc_c_.load_state_dict(best_ckpt['enc_c'])
    enc_d_.load_state_dict(best_ckpt['enc_d'])
    adapter_cell.load_state_dict(best_ckpt['ad_c'])
    adapter_drug.load_state_dict(best_ckpt['ad_d'])
    mlp_.load_state_dict(best_ckpt['mlp'])

    res = forward_predict(enc_c_, enc_d_, (adapter_cell, adapter_drug), mlp_, val_loader, device)
    # Post-RUS metadata from make_few_shot_loaders
    res["K_post"]     = int(info["K_post"])
    res["n_pos_post"] = int(info["n_pos_post"])
    res["n_neg_post"] = int(info["n_neg_post"])
    return res

# ====== Sweep ======
def few_shot_sweep_TCGA(
    few_shot_sizes: list, 
    run_id=0,
    epochs=25, 
    n_runs=5, 
    base_seed=123, 
    warmup_epochs=5, 
    freeze_drug_encoder=True,
    l2sp_lambda=1e-4,
    out_prefix="fewshot_tcga"
):
    """
    Run a sweep of few-shot adaptation experiments on TCGA over multiple K values.

    For each support size K, this function runs n_runs independent trials
    (different seeds), aggregates metrics (mean ± std), and outputs:
    - a CSV with aggregated results
    - a PNG plot of performance vs post-RUS support size (K_post)

    It also:
    - loads pretrained encoders and MLP head from disk
    - loads TCGA data and aligns TCGA features to the source feature columns
      saved in feature_columns.pkl
    - loads source train-time norms (cell/drug) for consistent normalization

    Parameters:
    - few_shot_sizes (list[int]): Explicit list of K support sizes to evaluate.
    - run_id (int); default=0; Which pretrained checkpoint to load (encoder_cell_trained_run_{run_id}.pt, etc.).
    - epochs (int); default=25; Adaptation epochs for Phase B inside run_one_few_shot.
    - n_runs (int); default=5; Number of random trials per K.
    - base_seed (int); default=123; Base seed; each run uses base_seed + r.
    - warmup_epochs (int); default=5; Warm-up epochs (adapters + MLP).
    - freeze_drug_encoder (bool); default=True: Freeze the drug encoder during partial unfreezing.
    - l2sp_lambda (float); default=1e-4; L2-SP strength.
    - out_prefix (str); default="fewshot_tcga"; Prefix used for saved output files.

    Return (pd.DataFrame): Aggregated sweep results with columns:
        K, AUC_mean, AUC_std, AUPRC_mean, AUPRC_std, K_post_mean, K_post_std.
    """
    cell_ae_latent_dim, drug_ae_latent_dim = 700, 50

    # Load pretrained encoders + MLP head
    encoder_cell = torch.load(f"encoder_cell_trained_run_{run_id}.pt", weights_only=False).to(device)
    encoder_drug = torch.load(f"encoder_drug_trained_run_{run_id}.pt", weights_only=False).to(device)
    mlp = MLP(cell_ae_latent_dim + drug_ae_latent_dim, 1)
    mlp.load_state_dict(torch.load(f"mlp_trained_run_{run_id}.pth"))
    mlp = mlp.to(device)

    # Load & align TCGA
    tcga_data, tcga_screen = RawDataLoader.load_data(
        data_modalities=DATA_MODALITIES,
        raw_file_directory=TCGA_DATA_FOLDER,
        screen_file_directory=TCGA_SCREENING_DATA,
        sep="\t"
    )

    with open("feature_columns.pkl", "rb") as f:
        all_features = pickle.load(f)
    for key in ['cell_exp', 'drug_desc', 'drug_finger']: # to adjust depending on modalities used during pretraining
        tcga_data[key] = tcga_data[key].reindex(columns=all_features[key], fill_value=0.0)
    x_cell, x_drug, y_tcga, _, _ = RawDataLoader.prepare_input_data(tcga_data, tcga_screen)
    y_np = np.asarray(y_tcga).ravel().astype(int)

    # --- Load train-time L2 norms ---
    cell_norms = torch.load(f"train_cell_l2norms_run_{run_id}.pt", map_location="cpu")
    drug_norms = torch.load(f"train_drug_l2norms_run_{run_id}.pt", map_location="cpu")
    
    # Guard tiny values exactly like the other scripts
    thr = 1e-6
    cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
    drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)

    # Build K list
    N = len(y_np)
    
    assert few_shot_sizes is not None and len(few_shot_sizes) > 0, \
    "few_shot_sizes must be provided"

    K_list = [int(k) for k in few_shot_sizes]

    if 0 not in K_list:
        K_list = [0] + K_list
        
    K_list = [max(0, min(k, N-2)) for k in K_list]

    metrics = ["AUC","AUPRC"]
    agg = {m: [] for m in metrics}; agg["K"] = []

    print(f"\n[Config] warmup_epochs={warmup_epochs} | freeze_drug={freeze_drug_encoder}\n")

    for K in K_list:
        print(f"--- K={K} over {n_runs} runs ---")
        run_vals = {m: [] for m in metrics}
        kposts = []
        for r in range(n_runs):
            seed = base_seed + r
            res = run_one_few_shot(
                x_cell, x_drug, y_np, K,
                encoder_cell, encoder_drug, mlp,
                cell_norms=cell_norms, drug_norms=drug_norms,
                epochs=epochs, batch_size=64,
                lr_ae=1e-5, lr_mlp=3e-4,
                seed=seed,
                warmup_epochs=warmup_epochs,
                freeze_drug_encoder=freeze_drug_encoder,
                l2sp_lambda=l2sp_lambda, run_id=run_id
            )
            for m in metrics:
                run_vals[m].append(float(res[m]))
            kposts.append(int(res.get("K_post", 0)))
        # mean ± std
        for m in metrics:
            mean, std = float(np.mean(run_vals[m])), float(np.std(run_vals[m], ddof=1) if n_runs > 1 else 0.0)
            agg[m].append((mean, std))
        agg["K"].append(K)
        agg.setdefault("K_post", []).append((
            float(np.mean(kposts)),
            float(np.std(kposts, ddof=1) if n_runs > 1 else 0.0)
        ))

    # Flatten to DataFrame
    rows = []
    for i, K in enumerate(agg["K"]):
        row = {"K": K}
        for m in metrics:
            row[m+"_mean"], row[m+"_std"] = agg[m][i][0], agg[m][i][1]
        row["K_post_mean"], row["K_post_std"] = agg["K_post"][i][0], agg["K_post"][i][1]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("K")

    # Plot
    os.makedirs("runs", exist_ok=True)
    tag = f"{out_prefix}_rid{run_id}"
    plt.figure(figsize=(6, 6))
    x_post = df["K_post_mean"].values
    
    for m in ["AUC","AUPRC"]:
        y  = df[m+"_mean"].values
        ys = df[m+"_std"].values
        plt.plot(x_post, y, marker='o', label=m)
        plt.fill_between(x_post, y-ys, y+ys, alpha=0.15)
    
    plt.title("Performance vs #TCGA samples (few-shot)\n(mean ± std over runs)")
    plt.xlabel("# TCGA samples (support size post RUS)")
    plt.ylabel("Metric")
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(ticks=x_post, labels=[str(int(round(k))) for k in x_post])
    
    plt.savefig(f"runs/{tag}.png", dpi=300, bbox_inches="tight")
    df.to_csv(f"runs/{tag}.csv", index=False)
    return df

if __name__ == "__main__":
    set_seed(123)
    df_full = few_shot_sweep_TCGA(
        run_id=0,
        few_shot_sizes=[0, 3, 6, 8, 11, 13, 16, 18, 20, 23, 26],
        epochs=25,
        n_runs=5,
        base_seed=123,
        warmup_epochs=5,
        freeze_drug_encoder=True,
        l2sp_lambda=1e-4,
        out_prefix="fewshot_tcga"
    )
    print("Few-shot model:\n", df_full)